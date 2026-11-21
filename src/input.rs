use std::fs::{self, File};
use std::path::Path;
use std::sync::{Arc, OnceLock, RwLock};

use ahash::AHashMap;
use memmap2::Mmap;
use rayon::prelude::*;

use crate::parse::{fields, lines, rows};
use crate::Result;

pub struct Inputs {
    maps: Vec<Mmap>,
}

impl Inputs {
    pub fn open(folder: &Path) -> Result<Self> {
        let mut paths = fs::read_dir(folder)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        paths.retain(|path| path.extension().is_some_and(|extension| extension == "csv"));
        paths.sort();
        let mut maps = Vec::new();
        for path in paths {
            let file = File::open(path)?;
            if file.metadata()?.len() > 0 {
                // /data is mounted read-only. Maps own the backing storage for
                // every borrowed ID until all processing and output finish.
                maps.push(unsafe { Mmap::map(&file)? });
            }
        }
        Ok(Self { maps })
    }

    pub fn bytes(&self) -> usize {
        self.maps.iter().map(|map| map.len()).sum()
    }

    pub fn chunks(&self) -> Vec<&[u8]> {
        self.chunks_with_size(4 * 1024 * 1024)
    }
    fn chunks_with_size(&self, chunk_bytes: usize) -> Vec<&[u8]> {
        let mut chunks = Vec::new();
        for map in &self.maps {
            // Each individual file has its own header, even header-only files.
            let mut start = memchr::memchr(b'\n', map).map_or(map.len(), |at| at + 1);
            while start < map.len() {
                let target = (start + chunk_bytes).min(map.len());
                let end = if target == map.len() {
                    target
                } else {
                    memchr::memchr(b'\n', &map[target..]).map_or(map.len(), |at| target + at + 1)
                };
                chunks.push(&map[start..end]);
                start = end;
            }
        }
        chunks
    }
}

const GENERIC: u64 = 1 << 63;
const INTERNED: u64 = 3 << 62;

// The most common representation stores a 'u' prefix and up to eight suffix
// bytes. A second range stores arbitrary short strings. A third refers to
// exact byte strings in the interner. A terminal NUL takes the interner path
// because the inline representation uses trailing zero padding.
#[inline]
fn inline_user(name: &[u8]) -> Option<u64> {
    if name.len() == 9 && name[0] == b'u' {
        let last = name[8];
        if last != 0 && last < 0x80 {
            return Some(u64::from_le_bytes(name[1..9].try_into().unwrap()));
        }
    }
    if name.first() == Some(&b'u') && name.len() <= 9 {
        let suffix = &name[1..];
        if suffix.last() == Some(&0) {
            return None;
        }
        let mut bytes = [0; 8];
        bytes[..suffix.len()].copy_from_slice(suffix);
        let word = u64::from_le_bytes(bytes);
        if word & GENERIC == 0 {
            return Some(word);
        }
    }
    if name.len() <= 7 && name.last() != Some(&0) {
        let mut bytes = [0; 8];
        bytes[..name.len()].copy_from_slice(name);
        return Some(GENERIC | u64::from_le_bytes(bytes));
    }
    None
}

#[derive(Default)]
struct LongNames {
    ids: AHashMap<Arc<[u8]>, u64>,
    names: Vec<Arc<[u8]>>,
}

impl LongNames {
    fn intern(&mut self, name: &[u8]) -> u64 {
        if let Some(key) = self.ids.get(name) {
            return *key;
        }
        let key = INTERNED | self.names.len() as u64;
        let name: Arc<[u8]> = name.into();
        self.names.push(name.clone());
        self.ids.insert(name, key);
        key
    }
}

// Inline generic keys occupy only the low 56 bits below their tag. This
// unused tag value cannot be produced by inline_user or the interner.
const VACANT: u64 = GENERIC | (1 << 60);

#[derive(Clone, Copy)]
struct Slot {
    key: u64,
    value: u64,
}

// Keys and values occupy the same cache line on a successful lookup.
// Linear probing also fetches neighbouring candidates on collisions.
struct AliasTable {
    // Four fingerprint bits per hash bucket. A clear bit proves that this
    // key is already canonical. A set bit still requires an exact lookup.
    redirected: Vec<u64>,
    slots: Vec<Slot>,
    shift: u32,
    len: usize,
}

impl AliasTable {
    #[inline]
    fn redirect_bit(&self, key: u64) -> usize {
        (crate::mix_key(key) >> (self.shift - 2)) as usize
    }

    #[inline]
    fn bucket(&self, key: u64) -> usize {
        let mixed = crate::mix_key(key);
        (mixed >> self.shift) as usize
    }

    #[inline]
    fn position(&self, key: u64) -> usize {
        let mut index = self.bucket(key);
        let mask = self.slots.len() - 1;
        loop {
            let candidate = self.slots[index].key;
            if candidate == VACANT || candidate == key {
                return index;
            }
            index = (index + 1) & mask;
        }
    }

    #[inline]
    fn get(&self, key: u64) -> u64 {
        let tag = self.redirect_bit(key);
        if self.redirected[tag >> 6] & (1u64 << (tag & 63)) == 0 {
            return key;
        }
        let slot = self.slots[self.position(key)];
        if slot.key == key {
            slot.value
        } else {
            key
        }
    }

    fn compress(&mut self, key: u64) -> u64 {
        let mut root = key;
        loop {
            let next = self.get(root);
            if next == root {
                break;
            }
            root = next;
        }
        let mut k = key;
        while k != root {
            let i = self.position(k);
            k = self.slots[i].value;
            self.slots[i].value = root;
        }
        root
    }
}

pub struct People {
    aliases: OnceLock<AliasTable>,
    long: RwLock<LongNames>,
}

impl People {
    pub fn new() -> Self {
        Self {
            aliases: OnceLock::new(),
            long: RwLock::new(LongNames::default()),
        }
    }

    pub fn load_links(&self, links: &Inputs) {
        // Each edge points from the greater spelling to the smaller one. Picking
        // one outgoing edge for each key immediately creates a decreasing forest.
        // Only keys with another distinct parent require a full union operation.
        let groups = if links.bytes() >= 1024 * 1024 {
            32usize
        } else {
            1
        };
        let shift = 64 - groups.trailing_zeros();
        let edges: Vec<_> = links
            .chunks_with_size(256 * 1024)
            .into_par_iter()
            .map(|chunk| {
                let mut buckets = (0..groups).map(|_| Vec::new()).collect::<Vec<_>>();
                for [a, b] in rows::<2>(chunk) {
                    if a.is_empty() || b.is_empty() || a == b {
                        continue;
                    }
                    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
                    let hi = self.key(hi);
                    let group = if groups == 1 {
                        0
                    } else {
                        (crate::mix_key(hi) >> shift) as usize
                    };
                    buckets[group].push((hi, self.key(lo)));
                }
                buckets
            })
            .collect();
        let count: usize = edges.iter().flatten().map(Vec::len).sum();
        let capacity = (count + count / 2 + 8).max(groups * 16).next_power_of_two();
        let shift = 64 - capacity.trailing_zeros();
        let width = capacity / groups;
        let mut aliases = AliasTable {
            slots: vec![
                Slot {
                    key: VACANT,
                    value: 0
                };
                capacity
            ],
            redirected: vec![0; capacity / 16],
            shift,
            len: 0,
        };
        let rest: Vec<_> = aliases
            .slots
            .par_chunks_mut(width)
            .zip(aliases.redirected.par_chunks_mut(width / 16))
            .enumerate()
            .map(|(group, (slots, bits))| {
                let mut overflow = Vec::new();
                let mut conflicts = Vec::new();
                let mut used = 0;
                for chunk in &edges {
                    for &(hi, lo) in &chunk[group] {
                        let h = crate::mix_key(hi);
                        let local = ((h >> shift) as usize) & (width - 1);
                        let bit = ((h >> (shift - 2)) as usize) & (4 * width - 1);
                        bits[bit >> 6] |= 1u64 << (bit & 63);
                        // Spill at the boundary so the completed table supports
                        // ordinary global linear probing for every later lookup.
                        match slots[local..]
                            .iter_mut()
                            .find(|s| s.key == VACANT || s.key == hi)
                        {
                            Some(s) if s.key == VACANT => {
                                *s = Slot { key: hi, value: lo };
                                used += 1;
                            }
                            Some(s) if s.value != lo => conflicts.push((s.value, lo)),
                            Some(_) => {}
                            None => overflow.push((hi, lo)),
                        }
                    }
                }
                (used, overflow, conflicts)
            })
            .collect();
        drop(edges);
        let mut conflicts = Vec::new();
        for (used, spill, mut more) in rest {
            aliases.len += used;
            conflicts.append(&mut more);
            for (hi, lo) in spill {
                let pos = aliases.position(hi);
                let s = &mut aliases.slots[pos];
                if s.key == VACANT {
                    *s = Slot { key: hi, value: lo };
                    aliases.len += 1;
                } else if s.value != lo {
                    conflicts.push((s.value, lo));
                }
            }
        }
        for (a, b) in conflicts {
            let left = aliases.compress(a);
            let right = aliases.compress(b);
            if left == right {
                continue;
            }
            let less = if (left & GENERIC == 0 && right & GENERIC == 0)
                || (left & INTERNED == GENERIC && right & INTERNED == GENERIC)
            {
                left.swap_bytes() < right.swap_bytes()
            } else {
                self.name(left) < self.name(right)
            };
            let (hi, lo) = if less { (right, left) } else { (left, right) };
            let at = aliases.position(hi);
            aliases.slots[at] = Slot { key: hi, value: lo };
            let bit = aliases.redirect_bit(hi);
            aliases.redirected[bit >> 6] |= 1u64 << (bit & 63);
            aliases.len += 1;
        }
        // Freeze short chains independently. Compressing the exceptional long
        // chains below bounds the cost of inputs arranged as a single long path.
        let roots: Vec<_> = aliases
            .slots
            .par_iter()
            .map(|slot| {
                if slot.key == VACANT {
                    return VACANT;
                }
                let mut key = slot.value;
                for _ in 0..5 {
                    let next = aliases.get(key);
                    if next == key {
                        return key;
                    }
                    key = next;
                }
                VACANT
            })
            .collect();
        for (i, root) in roots.into_iter().enumerate() {
            if aliases.slots[i].key != VACANT {
                let next = if root == VACANT {
                    aliases.compress(aliases.slots[i].value)
                } else {
                    root
                };
                aliases.slots[i].value = next;
            }
        }
        assert!(self.aliases.set(aliases).is_ok(), "links already loaded");
    }

    #[inline]
    pub fn key(&self, name: &[u8]) -> u64 {
        inline_user(name).unwrap_or_else(|| self.intern_long(name))
    }

    #[inline]
    pub fn canonical(&self, key: u64) -> u64 {
        self.aliases.get().expect("links must be loaded").get(key)
    }

    #[inline]
    pub fn resolve(&self, name: &[u8]) -> u64 {
        self.canonical(self.key(name))
    }

    #[cold]
    fn intern_long(&self, name: &[u8]) -> u64 {
        if let Some(key) = self.long.read().unwrap().ids.get(name) {
            return *key;
        }
        self.long.write().unwrap().intern(name)
    }

    pub fn with_name<T>(&self, key: u64, use_name: impl FnOnce(&[u8]) -> T) -> T {
        if key & INTERNED == INTERNED {
            let long = self.long.read().unwrap();
            return use_name(&long.names[(key & !INTERNED) as usize]);
        }
        let payload = key & !GENERIC;
        let len = 8 - (payload.leading_zeros() as usize / 8);
        let mut name = [0u8; 9];
        name[0] = b'u';
        name[1..].copy_from_slice(&payload.to_le_bytes());
        let start = usize::from(key & GENERIC != 0);
        use_name(&name[start..1 + len])
    }

    pub fn name(&self, key: u64) -> Vec<u8> {
        self.with_name(key, |name| name.to_vec())
    }

    pub fn len(&self) -> usize {
        self.aliases.get().map_or(0, |aliases| aliases.len)
    }
}

struct DimensionIndex {
    prefix: &'static [u8],
    dense: Vec<u32>,
    other: AHashMap<Vec<u8>, u32>,
}

impl DimensionIndex {
    fn new(prefix: &'static [u8], rows: usize) -> Self {
        Self {
            prefix,
            dense: vec![u32::MAX; rows + 1],
            other: AHashMap::new(),
        }
    }

    // Only the canonical ASCII decimal spelling can index the array. IDs
    // with leading zeros or arbitrary bytes use exact byte matching.
    #[inline]
    fn slot(&self, name: &[u8]) -> Option<usize> {
        let value = name.strip_prefix(self.prefix)?;
        if value.is_empty() || value.len() > 8 || (value.len() > 1 && value[0] == b'0') {
            return None;
        }
        let mut index = 0;
        for byte in value {
            let digit = byte.wrapping_sub(b'0');
            if digit > 9 {
                return None;
            }
            index = 10 * index + usize::from(digit);
        }
        (index < self.dense.len()).then_some(index)
    }

    fn insert(&mut self, name: &[u8], value: u32) {
        if let Some(index) = self.slot(name) {
            self.dense[index] = value;
        } else {
            self.other.insert(name.to_vec(), value);
        }
    }

    #[inline]
    fn get(&self, name: &[u8]) -> Option<&u32> {
        if let Some(index) = self.slot(name) {
            let value = &self.dense[index];
            (*value != u32::MAX).then_some(value)
        } else {
            self.other.get(name)
        }
    }
}

pub struct Dimensions {
    products: DimensionIndex,
    pub categories: Vec<Vec<u8>>,
    pub unknown: u32,
    campaigns: DimensionIndex,
    pub campaign_names: Vec<Vec<u8>>,
    pub channels: Vec<Vec<u8>>,
    pub organic: u32,
}

impl Dimensions {
    #[inline]
    pub fn category_for(&self, product: &[u8]) -> u32 {
        self.products.get(product).copied().unwrap_or(self.unknown)
    }

    #[inline]
    pub fn campaign_for(&self, campaign: &[u8]) -> Option<u32> {
        self.campaigns.get(campaign).copied()
    }

    pub fn load(data: &Path) -> Result<Self> {
        let product_bytes = fs::read(data.join("products.csv"))?;
        let product_rows: Vec<_> = lines(&product_bytes)
            .skip(1)
            .filter_map(fields::<2>)
            .collect();
        let mut labels: Vec<&[u8]> = product_rows.iter().map(|row| row[1]).collect();
        labels.push(b"unknown");
        labels.sort_unstable();
        labels.dedup();
        let unknown = labels.binary_search(&b"unknown".as_slice()).unwrap() as u32;
        let mut products = DimensionIndex::new(b"p", product_rows.len());
        for [product, category] in product_rows {
            products.insert(product, labels.binary_search(&category).unwrap() as u32);
        }
        let categories = labels.into_iter().map(Vec::from).collect();
        let campaign_bytes = fs::read(data.join("campaigns.csv"))?;
        let campaign_rows: Vec<_> = lines(&campaign_bytes)
            .skip(1)
            .filter_map(fields::<2>)
            .collect();
        let mut campaigns = DimensionIndex::new(b"c_", campaign_rows.len());
        let mut campaign_names = Vec::new();
        let mut channels = Vec::new();
        for [campaign, channel] in campaign_rows {
            campaigns.insert(campaign, campaign_names.len() as u32);
            campaign_names.push(campaign.to_vec());
            channels.push(channel.to_vec());
        }
        let organic = campaign_names.len() as u32;
        campaign_names.push(b"organic".to_vec());
        channels.push(b"organic".to_vec());
        Ok(Self {
            products,
            categories,
            unknown,
            campaigns,
            campaign_names,
            channels,
            organic,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decreasing_forest_handles_conflicting_parents_and_cluster_wraparound() {
        let people = People::new();
        let keys: Vec<_> = (0..200_000u32)
            .map(|n| format!("u{n:08x}"))
            .filter(|name| crate::mix_key(people.key(name.as_bytes())) >> 55 == 511)
            .take(10)
            .collect();
        assert_eq!(keys.len(), 10);

        let mut csv = b"user_id_a,user_id_b\n".to_vec();
        for key in &keys {
            csv.extend_from_slice(format!("{key},a\n").as_bytes());
        }
        csv.extend_from_slice(b"z,q\nz,r\nr,b\nq,z\nextra-long-identity,z\n");
        for n in 0..32 {
            csv.extend_from_slice(format!("n{n:02},n{:02}\n", n + 1).as_bytes());
        }
        // One malformed line selects the multi-partition path without changing
        // the graph or its 512-slot capacity.
        csv.resize(csv.len() + 1024 * 1024, b'x');
        let mut map = memmap2::MmapMut::map_anon(csv.len()).unwrap();
        map.copy_from_slice(&csv);
        people.load_links(&Inputs {
            maps: vec![map.make_read_only().unwrap()],
        });
        for key in keys {
            assert_eq!(people.name(people.resolve(key.as_bytes())), b"a");
        }
        for name in [b"z".as_slice(), b"r", b"q", b"extra-long-identity"] {
            assert_eq!(people.name(people.resolve(name)), b"b");
        }
        assert_eq!(people.name(people.resolve(b"n32")), b"n00");
        assert_eq!(people.name(people.resolve(b"unlinked")), b"unlinked");
    }

    #[test]
    fn redirect_fingerprint_collision_still_checks_the_key() {
        let mut aliases = AliasTable {
            slots: vec![
                Slot {
                    key: VACANT,
                    value: 0
                };
                16
            ],
            redirected: vec![0],
            len: 0,
            shift: 60,
        };
        let raw = 9;
        let canonical = 3;
        let index = aliases.position(raw);
        aliases.slots[index].key = raw;
        aliases.slots[index].value = canonical;
        let bit = aliases.redirect_bit(raw);
        aliases.redirected[0] |= 1 << bit;
        let collision = (raw + 1..)
            .find(|&key| aliases.redirect_bit(key) == bit)
            .unwrap();
        assert_eq!(aliases.get(raw), canonical);
        assert_eq!(aliases.get(collision), collision);
        assert_eq!(aliases.get(canonical), canonical);
    }
    use std::collections::HashSet;

    #[test]
    fn user_keys_roundtrip_exact_bytes_in_all_representations() {
        let people = People::new();
        let values: &[&[u8]] = &[
            b"",
            b"u",
            b"u\0",
            b"u\0a",
            b"a",
            b"a\0",
            b"u12345678",
            b"u1234567\0",
            b"u12345678\0",
            b"1234567",
            b"12345678",
            b"u1234567\x80",
            "雪".as_bytes(),
            "u雪雪雪".as_bytes(),
        ];
        let mut keys = HashSet::new();
        for &value in values {
            let key = people.key(value);
            assert!(keys.insert(key));
            assert_eq!(people.key(value), key);
            assert_eq!(people.name(key), value);
        }
    }

    #[test]
    fn dense_dimension_slots_keep_noncanonical_names_distinct() {
        let mut campaigns = DimensionIndex::new(b"c_", 4);
        for (index, name) in [b"c_2".as_slice(), b"c_02", b"c_22", b"other", b"c_2\0"]
            .iter()
            .enumerate()
        {
            campaigns.insert(name, index as u32);
            assert_eq!(campaigns.get(name), Some(&(index as u32)));
        }
        assert_eq!(campaigns.get(b"c_2"), Some(&0));
        assert_eq!(campaigns.get(b"c_1"), None);
        assert_eq!(campaigns.get(b"c_0002"), None);
    }
}
