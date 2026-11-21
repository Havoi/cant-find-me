use ahash::AHashMap;
use rayon::prelude::*;

use crate::input::Inputs;
use crate::parse::{rows, unsigned, TimestampParser, DAY};
use crate::{
    buckets, transpose, Ad, ClickTime, Context, GroupEvent, KeyedClick, KeyedEvent, PersonEvent,
};

// Each shard-local key record names a payload in its own CSV chunk. Payloads
// are partitioned by the raw person's alias-table bucket for the later lookup.
#[derive(Clone, Copy)]
struct Entry {
    key: u64,
    location: u32,
    ingest: u32,
}

// One allocation supplies the usual capacity of every destination in a
// source chunk. A busy destination appends to its own spill vector; its
// source indexes continue from the last primary row.
struct ChunkRows<T: Copy> {
    primary: Vec<std::mem::MaybeUninit<T>>,
    width: usize,
    counts: Vec<usize>,
    spills: Vec<Vec<T>>,
}

impl<T: Copy> ChunkRows<T> {
    fn new(groups: usize, estimated_per_group: usize) -> Self {
        // An odd cache-line stride spreads equal positions in consecutive
        // partitions across cache sets. Both stored record types are 16 bytes.
        const {
            assert!(std::mem::size_of::<T>() == 16);
        }
        let width = if estimated_per_group == 0 {
            0
        } else {
            (estimated_per_group.div_ceil(4) | 1) * 4
        };
        let slots = width.checked_mul(groups).expect("too many rows in a chunk");
        let mut primary = Vec::with_capacity(slots);
        // Each shard initializes its prefix in insertion order before it is read.
        primary.resize_with(slots, std::mem::MaybeUninit::uninit);
        Self {
            primary,
            width,
            counts: vec![0; groups],
            spills: buckets(groups),
        }
    }

    #[inline]
    fn push(&mut self, group: usize, row: T) -> usize {
        let count = &mut self.counts[group];
        let index = *count;
        *count += 1;
        if index < self.width {
            // counts indexed the same group above and group * width < slots.
            unsafe {
                self.primary
                    .get_unchecked_mut(group * self.width + index)
                    .write(row);
            }
        } else {
            self.spills[group].push(row);
        }
        index
    }

    fn take_shards(&mut self) -> Vec<Rows<'_, T>> {
        let mut left = self.primary.as_mut_slice();
        self.counts
            .iter()
            .zip(&mut self.spills)
            .map(|(&count, spill)| {
                let (section, rest) = std::mem::take(&mut left).split_at_mut(self.width);
                left = rest;
                let initialized = count.min(self.width);
                // Only these initialized rows are exposed. split_at_mut proves
                // the shard slices do not overlap before Rayon distributes them.
                let head = unsafe {
                    std::slice::from_raw_parts_mut(section.as_mut_ptr().cast::<T>(), initialized)
                };
                Rows {
                    head,
                    tail: std::mem::take(spill),
                }
            })
            .collect()
    }
}

struct Rows<'a, T> {
    head: &'a mut [T],
    tail: Vec<T>,
}

impl<T> Default for Rows<'_, T> {
    fn default() -> Self {
        Self {
            head: &mut [],
            tail: Vec::new(),
        }
    }
}

impl<T> Rows<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        self.head.len() + self.tail.len()
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = &T> {
        self.head.iter().chain(&self.tail)
    }

    #[cfg(test)]
    fn push(&mut self, row: T) {
        self.tail.push(row)
    }
}

impl<T> std::ops::Index<usize> for Rows<'_, T> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &T {
        if index < self.head.len() {
            &self.head[index]
        } else {
            &self.tail[index - self.head.len()]
        }
    }
}

impl<T> std::ops::IndexMut<usize> for Rows<'_, T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        if index < self.head.len() {
            &mut self.head[index]
        } else {
            &mut self.tail[index - self.head.len()]
        }
    }
}

const ABSENT: u32 = u32::MAX;
const LOCATION_BITS: usize = 8;
const GROUP_MASK: usize = 127;
const WIDE: usize = 128;
const EXTENDED_INGEST: u32 = u32::MAX;

#[derive(Default)]
struct IdPart<'a> {
    entries: Rows<'a, Entry>,
    extended: AHashMap<usize, i64>,
    general: Vec<(&'a [u8], i64, u32)>,
}

impl IdPart<'_> {
    #[inline]
    fn ingestion(&self, index: usize) -> i64 {
        let ordinary = self.entries[index].ingest;
        if ordinary == EXTENDED_INGEST {
            self.extended[&index]
        } else {
            i64::from(ordinary) * 1000
        }
    }
}

#[derive(Clone, Copy)]
struct Packed {
    person: u64,
    offset: u32,
    finance: u32,
}

#[derive(Default)]
struct PersonPart<'a> {
    packed: Rows<'a, Packed>,
    wide: Vec<PersonEvent>,
}

#[derive(Default)]
struct IdDetails<'a> {
    extended: AHashMap<usize, i64>,
    general: Vec<(&'a [u8], i64, u32)>,
}

struct IdChunk<'a> {
    rows: ChunkRows<Entry>,
    details: Vec<IdDetails<'a>>,
}

struct PersonChunk {
    rows: ChunkRows<Packed>,
    wide: Vec<Vec<PersonEvent>>,
}

pub(crate) struct EventParts<'a> {
    ids: Vec<IdChunk<'a>>,
    people: Vec<PersonChunk>,
}

// With a shared leading byte, up to eight remaining bytes fit in one word.
// Requiring the last byte to be nonzero makes zero padding unambiguous.
// Other strings retain their exact bytes in a separate table.
#[inline]
fn compact_key(id: &[u8], prefix: u8) -> Option<u64> {
    if id.first() != Some(&prefix) || id.len() < 2 || id.len() > 9 || id.last() == Some(&0) {
        return None;
    }
    let suffix = &id[1..];
    let word = if suffix.len() == 8 {
        u64::from_le_bytes(suffix.try_into().unwrap())
    } else {
        let mut bytes = [0; 8];
        bytes[..suffix.len()].copy_from_slice(suffix);
        u64::from_le_bytes(bytes)
    };
    // Xor shifts and odd multiplication are bijections, so equality of mixed
    // words is exact even when IDs share a hash bucket. Zero stays reserved.
    let mixed = crate::mix_key(word);
    Some(mixed ^ (mixed >> 32))
}

#[inline]
fn alias_group(person: u64, groups: usize) -> usize {
    let mixed = crate::mix_key(person);
    ((mixed >> (64 - 7)) as usize) & (groups - 1)
}

#[inline]
fn save_payload(parts: &mut PersonChunk, event: PersonEvent, context: &Context) -> u32 {
    let group = alias_group(event.person, parts.wide.len());
    let offset = event.ts - context.start;
    let index = if offset < i64::from(u32::MAX) && (event.revenue == 0 || event.category <= 255) {
        let index = parts.rows.push(
            group,
            Packed {
                person: event.person,
                offset: offset as u32,
                finance: if event.revenue == 0 {
                    0
                } else {
                    (event.category << 24) | event.revenue
                },
            },
        );
        index << LOCATION_BITS
    } else {
        let bucket = &mut parts.wide[group];
        let index = (bucket.len() << LOCATION_BITS) | WIDE;
        bucket.push(event);
        index
    };
    // Inputs::chunks is at most 4 MiB plus one terminal line. Even the
    // shortest 7-field input record consumes seven bytes.
    u32::try_from(index | group).expect("too many records in a chunk")
}

enum EventId<'a> {
    Compact(u64),
    Other(&'a [u8]),
}

pub(crate) fn scan_events<'a>(input: &'a Inputs, context: &Context) -> EventParts<'a> {
    let chunks = input.chunks();
    let prefix = chunks
        .first()
        .into_iter()
        .flat_map(|chunk| rows::<7>(chunk))
        .find_map(|row| row[0].first().copied())
        .unwrap_or(0);
    let group_count = context.event_shards.min(32);
    let chunks: Vec<_> = chunks
        .par_iter()
        .map(|chunk| {
            let estimated = chunk.len() / 76;
            let mut ids = IdChunk {
                rows: ChunkRows::new(
                    context.event_shards,
                    estimated * 11 / 10 / context.event_shards,
                ),
                details: (0..context.event_shards)
                    .map(|_| IdDetails::default())
                    .collect(),
            };
            let mut people = PersonChunk {
                rows: ChunkRows::new(group_count, estimated * 11 / 10 / group_count),
                wide: buckets(group_count),
            };
            let mut dates = TimestampParser::new();
            crate::parse::visit_event_rows(chunk, prefix, |row| {
                let (id, seconds, location) = match row {
                    crate::parse::EventInput::Activity {
                        suffix,
                        person,
                        timestamp,
                        arrival,
                    } => {
                        let (seconds, timestamp) = dates.event_arrival(timestamp, arrival);
                        let location = timestamp
                            .filter(|&ts| {
                                ts >= context.start
                                    && ts < context.end
                                    && seconds * 1000 - ts <= DAY
                            })
                            .map_or(ABSENT, |ts| {
                                save_payload(
                                    &mut people,
                                    PersonEvent {
                                        person,
                                        ts,
                                        revenue: 0,
                                        category: context.dimensions.unknown,
                                    },
                                    context,
                                )
                            });
                        let mixed = crate::mix_key(suffix);
                        (EventId::Compact(mixed ^ (mixed >> 32)), seconds, location)
                    }
                    crate::parse::EventInput::General(
                        [id, user, ts, ingest, kind, product, price],
                    ) => {
                        if id.is_empty() {
                            return;
                        }
                        let seconds = dates.ingest_seconds(ingest);
                        let ingest = seconds * 1000;
                        let revenue = if kind == b"purchase" {
                            unsigned(price)
                                .filter(|value| (1..=10_000_000).contains(value))
                                .unwrap_or(0) as u32
                        } else {
                            0
                        };
                        let valid = !user.is_empty()
                            && !product.is_empty()
                            && (revenue > 0 || kind == b"view" || kind == b"cart");
                        let location = if let Some(ts) = dates.timestamp(ts).filter(|&ts| {
                            valid && ts >= context.start && ts < context.end && ingest - ts <= DAY
                        }) {
                            let category = if revenue == 0 {
                                context.dimensions.unknown
                            } else {
                                context.dimensions.category_for(product)
                            };
                            save_payload(
                                &mut people,
                                PersonEvent {
                                    person: context.people.key(user),
                                    ts,
                                    revenue,
                                    category,
                                },
                                context,
                            )
                        } else {
                            ABSENT
                        };
                        let id =
                            compact_key(id, prefix).map_or(EventId::Other(id), EventId::Compact);
                        (id, seconds, location)
                    }
                };
                let ingest = seconds * 1000;
                let mask = context.event_shards - 1;
                match id {
                    EventId::Compact(key) => {
                        let shard = key as usize & mask;
                        let seconds = seconds as u64;
                        let ingest = if seconds < u64::from(EXTENDED_INGEST) {
                            seconds as u32
                        } else {
                            ids.details[shard]
                                .extended
                                .insert(ids.rows.counts[shard], ingest);
                            EXTENDED_INGEST
                        };
                        ids.rows.push(
                            shard,
                            Entry {
                                key,
                                location,
                                ingest,
                            },
                        );
                    }
                    EventId::Other(id) => {
                        let shard = context.hash.hash_one(id) as usize & mask;
                        ids.details[shard].general.push((id, ingest, location));
                    }
                }
            });
            (ids, people)
        })
        .collect();
    let (ids, people) = chunks.into_iter().unzip();
    EventParts { ids, people }
}

#[inline]
fn reject(losers: &mut [Vec<u64>], chunk: usize, location: u32) {
    if location != ABSENT {
        losers[location as usize & GROUP_MASK].push(((chunk as u64) << 32) | u64::from(location));
    }
}

fn rejected(parts: Vec<IdPart<'_>>, groups: usize) -> Vec<Vec<u64>> {
    let count: usize = parts.iter().map(|part| part.entries.len()).sum();
    let max_len = parts
        .iter()
        .map(|part| part.entries.len())
        .max()
        .unwrap_or(0);
    let row_bits = usize::BITS - max_len.leading_zeros();
    let row_mask = (1u64 << row_bits) - 1;
    // One slot stores a quick key fingerprint and the source location. A
    // fingerprint match always checks the entire original key before merging.
    // A slot observed once proves its full event ID is unique. Collisions
    // and real duplicates enter the owner table and compare the original key.
    let bit_count = (count * 8).max(64).next_power_of_two();
    let mask = bit_count - 1;
    let mut seen = vec![0u64; bit_count / 64];
    let mut repeated = vec![0u64; seen.len()];
    for part in &parts {
        for entry in part.entries.iter() {
            let n = (entry.key >> 29) as usize & mask;
            let bit = 1u64 << (n & 63);
            repeated[n >> 6] |= seen[n >> 6] & bit;
            seen[n >> 6] |= bit;
        }
    }
    let singles: usize = seen
        .iter()
        .zip(&repeated)
        .map(|(&s, &r)| (s & !r).count_ones() as usize)
        .sum();
    let suspected = count - singles;
    drop(seen);
    let capacity = (suspected + suspected / 2 + 1).next_power_of_two();
    let mut slots = vec![0u64; capacity];
    let mut losers = buckets(groups);
    let mut general = AHashMap::new();
    if parts.len() as u64 > (u64::from(u32::MAX) >> row_bits) {
        // A huge number of tiny input files may make a sparse (chunk, row)
        // position wider than the in-slot representation.
        let mut exact = AHashMap::new();
        for (chunk, part) in parts.iter().enumerate() {
            for (row, entry) in part.entries.iter().enumerate() {
                let next = (part.ingestion(row), chunk, entry.location);
                match exact.entry(entry.key) {
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        slot.insert(next);
                    }
                    std::collections::hash_map::Entry::Occupied(mut slot) => {
                        let old = *slot.get();
                        if next.0 < old.0 {
                            reject(&mut losers, old.1, old.2);
                            slot.insert(next);
                        } else {
                            reject(&mut losers, next.1, next.2);
                        }
                    }
                }
            }
        }
    } else {
        for (chunk, part) in parts.iter().enumerate() {
            for (row, &entry) in part.entries.iter().enumerate() {
                let bit = (entry.key >> 29) as usize & mask;
                if repeated[bit >> 6] & (1u64 << (bit & 63)) == 0 {
                    continue;
                }
                let owner = (((chunk as u64) << row_bits) | row as u64) + 1;
                let fingerprint = entry.key & 0xffff_ffff_0000_0000;
                let encoded = fingerprint | owner;
                let mut slot = (entry.key >> 10) as usize & (capacity - 1);
                loop {
                    let prior = slots[slot];
                    if prior == 0 {
                        slots[slot] = encoded;
                        break;
                    }
                    if (prior ^ encoded) >> 32 == 0 {
                        let previous = (prior as u32 - 1) as u64;
                        let old_chunk = (previous >> row_bits) as usize;
                        let old_row = (previous & row_mask) as usize;
                        let old = parts[old_chunk].entries[old_row];
                        if entry.key == old.key {
                            let earlier = if entry.ingest != EXTENDED_INGEST
                                && old.ingest != EXTENDED_INGEST
                            {
                                entry.ingest < old.ingest
                            } else {
                                part.ingestion(row) < parts[old_chunk].ingestion(old_row)
                            };
                            if earlier {
                                reject(&mut losers, old_chunk, old.location);
                                slots[slot] = encoded;
                            } else {
                                reject(&mut losers, chunk, entry.location);
                            }
                            break;
                        }
                    }
                    slot = (slot + 1) & (capacity - 1);
                }
            }
        }
    }
    for (chunk, part) in parts.into_iter().enumerate() {
        for (id, ingest, location) in part.general {
            let next = (ingest, chunk, location);
            match general.entry(id) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(next);
                }
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    let old = *slot.get();
                    if ingest < old.0 {
                        reject(&mut losers, old.1, old.2);
                        slot.insert(next);
                    } else {
                        reject(&mut losers, chunk, location);
                    }
                }
            }
        }
    }
    losers
}

fn resolve<T: GroupEvent>(
    mut parts: Vec<PersonPart>,
    losses: Vec<Vec<u64>>,
    context: &Context,
) -> Vec<Vec<KeyedEvent<T>>> {
    for batch in losses {
        for rejected in batch {
            let chunk = (rejected >> 32) as usize;
            let location = rejected as u32 as usize;
            let index = location >> LOCATION_BITS;
            if location & WIDE == 0 {
                parts[chunk].packed[index].offset = u32::MAX;
            } else {
                parts[chunk].wide[index].ts = i64::MIN;
            }
        }
    }
    let expected: usize = parts
        .iter()
        .map(|part| part.packed.len() + part.wide.len())
        .sum();
    let mut output: Vec<_> = (0..context.person_shards)
        .map(|_| Vec::with_capacity(expected / context.person_shards))
        .collect();
    for part in parts {
        let packed = part
            .packed
            .iter()
            .filter(|row| row.offset != u32::MAX)
            .map(|row| {
                (
                    row.person,
                    T::from_packed(row.offset, row.finance, context.start),
                )
            });
        let wide = part
            .wide
            .into_iter()
            .filter(|row| row.ts != i64::MIN)
            .map(|row| (row.person, T::pack(row, context.start)));
        for (raw, data) in packed.chain(wide) {
            let person = context.people.canonical(raw);
            output[context.person_shard(person)].push(KeyedEvent { person, data });
        }
    }
    output
}

pub(crate) fn finish_events<T: GroupEvent>(
    raw: EventParts<'_>,
    context: &Context,
) -> Vec<Vec<Vec<KeyedEvent<T>>>> {
    let EventParts {
        mut ids,
        mut people,
    } = raw;
    let groups = context.event_shards.min(32);
    let mut id_shards = buckets(context.event_shards);
    for chunk in &mut ids {
        let details = std::mem::take(&mut chunk.details);
        for (shard, (entries, info)) in id_shards
            .iter_mut()
            .zip(chunk.rows.take_shards().into_iter().zip(details))
        {
            shard.push(IdPart {
                entries,
                extended: info.extended,
                general: info.general,
            });
        }
    }
    let losses = id_shards
        .into_par_iter()
        .map(|parts| rejected(parts, groups))
        .collect();
    drop(ids);
    let losses = transpose(losses, groups);
    let mut people_shards = buckets(groups);
    for chunk in &mut people {
        let wide = std::mem::take(&mut chunk.wide);
        for (shard, (packed, wide)) in people_shards
            .iter_mut()
            .zip(chunk.rows.take_shards().into_iter().zip(wide))
        {
            shard.push(PersonPart { packed, wide });
        }
    }
    // Distribute the largest alias groups before workers take smaller ones.
    let mut jobs: Vec<_> = people_shards
        .into_iter()
        .zip(losses)
        .map(|(parts, losses)| {
            let count = parts
                .iter()
                .map(|part| part.packed.len() + part.wide.len())
                .sum::<usize>();
            (count, parts, losses)
        })
        .collect();
    jobs.sort_unstable_by_key(|job| std::cmp::Reverse(job.0));
    let result = jobs
        .into_iter()
        .par_bridge()
        .map(|(_, parts, losses)| resolve::<T>(parts, losses, context))
        .collect();
    transpose(result, context.person_shards)
}

pub(crate) fn clicks<C: ClickTime>(context: &Context) -> Vec<Vec<Vec<KeyedClick<C>>>> {
    let origin = context.click_origin();
    let chunks = &context.click_sources;
    let groups: Vec<_> = chunks
        .par_iter()
        .enumerate()
        .map(|(source, chunk)| {
            let mut groups: Vec<_> = (0..context.person_shards)
                .map(|_| Vec::with_capacity(chunk.len() / 57 / context.person_shards))
                .collect();
            let mut dates = TimestampParser::new();
            for [id, user, campaign, ts] in rows::<4>(chunk) {
                if id.is_empty() || user.is_empty() {
                    continue;
                }
                let Some(campaign) = context.dimensions.campaign_for(campaign) else {
                    continue;
                };
                let Some(ts) = dates
                    .timestamp(ts)
                    .filter(|&ts| ts >= origin && ts < context.end)
                else {
                    continue;
                };
                let person = context.person(user);
                let id = (u64::from(u32::try_from(source).expect("too many click input chunks"))
                    << 32)
                    | u64::from(
                        u32::try_from(id.as_ptr() as usize - chunk.as_ptr() as usize)
                            .expect("click row start exceeds chunk size"),
                    );
                groups[context.person_shard(person)].push(KeyedClick {
                    person,
                    click: Ad {
                        id,
                        time: C::encode(ts, origin),
                        campaign,
                    },
                });
            }
            groups
        })
        .collect();
    transpose(groups, context.person_shards)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_encoding_is_exact_and_all_other_ids_have_a_fallback() {
        let values: &[&[u8]] = &[
            b"e1",
            b"e01",
            b"e\x001",
            b"e1\x001",
            b"e12345678",
            b"e12345679",
            b"e\xff",
        ];
        let keys: ahash::AHashSet<_> = values
            .iter()
            .map(|value| compact_key(value, b'e').unwrap())
            .collect();
        assert_eq!(keys.len(), values.len());
        assert!(!keys.contains(&0));
        for id in [&b"e"[..], b"e1\0", b"x1234", b"e123456789"] {
            assert!(compact_key(id, b'e').is_none());
        }
    }

    #[test]
    fn primary_ranges_and_spills_preserve_every_logical_position() {
        for initial in [0, 1, 5, 16] {
            let mut storage = ChunkRows::new(4, initial);
            let mut expected = vec![vec![]; 4];
            for i in 0..95u64 {
                let group = (i * i % 3) as usize;
                let row = [i, !i];
                assert_eq!(storage.push(group, row), expected[group].len());
                expected[group].push(row);
            }
            for (mut rows, values) in storage.take_shards().into_iter().zip(expected) {
                assert!(rows.iter().copied().eq(values.iter().copied()));
                for (index, value) in values.into_iter().enumerate() {
                    assert_eq!(rows[index], value);
                    rows[index][0] += 1;
                    assert_eq!(rows[index][0], value[0] + 1);
                }
            }
        }
    }

    #[test]
    fn repeated_slot_checks_full_id_and_keeps_the_earliest_copy() {
        let (first, second) = (0x7000_0000_0000_0001u64, 0x7000_0000_0000_0002u64);
        // The keys share both the bitmap slot and the entire owner fingerprint.
        // Chunk 1 has an invalid earlier copy of first and a late copy of second.
        let mut a = IdPart::default();
        a.entries.push(Entry {
            key: first,
            location: 0,
            ingest: 15,
        });
        a.entries.push(Entry {
            key: second,
            location: 1 << LOCATION_BITS,
            ingest: EXTENDED_INGEST,
        });
        a.extended.insert(1, -1000);
        let mut b = IdPart::default();
        b.entries.push(Entry {
            key: first,
            location: ABSENT,
            ingest: 9,
        });
        b.entries.push(Entry {
            key: second,
            location: 0,
            ingest: 11,
        });
        a.general
            .push((b"long arbitrary id", 5_000, 2 << LOCATION_BITS));
        b.general
            .push((b"long arbitrary id", 8_000, 1 << LOCATION_BITS));
        let mut losers = rejected(vec![a, b], 1);
        losers[0].sort_unstable();
        assert_eq!(
            losers[0],
            [0, 1u64 << 32, (1u64 << 32) | (1 << LOCATION_BITS)]
        );
    }
}
