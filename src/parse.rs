use memchr::{memchr, memchr_iter};

pub const DAY: i64 = 86_400_000;
pub const WEEK: i64 = 7 * DAY;
pub const SESSION_GAP: i64 = 1_800_000;

/// The challenge permits quotes only around a complete field and never escapes.
pub fn fields<const N: usize>(line: &[u8]) -> Option<[&[u8]; N]> {
    let mut result = [&[][..]; N];
    let mut start = 0;
    let mut count = 0;
    for end in memchr_iter(b',', line).chain(std::iter::once(line.len())) {
        if count == N {
            return None;
        }
        let mut field = &line[start..end];
        if field.first() == Some(&b'"') {
            if field.len() < 2 || field.last() != Some(&b'"') {
                return None;
            }
            field = &field[1..field.len() - 1];
        }
        if memchr(b'"', field).is_some() {
            return None;
        }
        result[count] = field;
        count += 1;
        start = end + 1;
    }
    (count == N).then_some(result)
}

pub fn lines(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut start = 0;
    memchr_iter(b'\n', bytes)
        .chain((!bytes.is_empty() && bytes.last() != Some(&b'\n')).then_some(bytes.len()))
        .map(move |end| {
            let line = &bytes[start..end];
            start = end + 1;
            line
        })
}

/// Each mask bit classifies the corresponding byte of a complete 64-byte block.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "bmi2"
))]
#[derive(Default)]
struct Structural {
    separators: u64,
    newlines: u64,
    quotes: u64,
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "bmi2"
))]
#[inline(always)]
#[cfg(not(target_feature = "avx512bw"))]
fn classify(block: &[u8; 64]) -> Structural {
    let a = classify32(block.first_chunk().unwrap());
    let b = classify32(block[32..].first_chunk().unwrap());
    Structural {
        newlines: a.newlines | (b.newlines << 32),
        separators: a.separators | (b.separators << 32),
        quotes: a.quotes | (b.quotes << 32),
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "bmi2",
    target_feature = "avx512bw"
))]
#[inline(always)]
fn classify(block: &[u8; 64]) -> Structural {
    use std::arch::x86_64::*;
    // The caller has checked all 64 bytes, including unaligned input.
    unsafe {
        let bytes = _mm512_loadu_si512(block.as_ptr().cast());
        let newlines = _mm512_cmpeq_epi8_mask(bytes, _mm512_set1_epi8(b'\n' as i8));
        let commas = _mm512_cmpeq_epi8_mask(bytes, _mm512_set1_epi8(b',' as i8));
        let quotes = _mm512_cmpeq_epi8_mask(bytes, _mm512_set1_epi8(b'"' as i8));
        Structural {
            separators: commas | newlines,
            newlines,
            quotes,
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "bmi2"
))]
#[inline(always)]
fn classify_second(block: &[u8; 64]) -> Structural {
    #[cfg(target_feature = "avx512bw")]
    {
        classify(block)
    }
    #[cfg(not(target_feature = "avx512bw"))]
    {
        let mut first = classify32(block.first_chunk().unwrap());
        if first.newlines == 0 {
            let next = classify32(block[32..].first_chunk().unwrap());
            first.newlines |= next.newlines << 32;
            first.separators |= next.separators << 32;
            first.quotes |= next.quotes << 32;
        }
        first
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "bmi2"
))]
#[inline(always)]
fn classify32(block: &[u8; 32]) -> Structural {
    use std::arch::x86_64::*;
    // The 32-byte load is within the block and AVX2 is enabled.
    unsafe {
        let input = _mm256_loadu_si256(block.as_ptr().cast());
        let matches = |byte| {
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(input, _mm256_set1_epi8(byte))) as u32 as u64
        };
        let newlines = matches(b'\n' as i8);
        Structural {
            newlines,
            separators: newlines | matches(b',' as i8),
            quotes: matches(b'"' as i8),
        }
    }
}

/// Iterate over well-formed rows, preserving byte-exact field values.
pub fn rows<const N: usize>(data: &[u8]) -> FastRows<'_, N> {
    FastRows { data }
}

pub struct FastRows<'a, const N: usize> {
    data: &'a [u8],
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "bmi2"
))]
#[inline(always)]
fn fast_record<const N: usize>(line: &[u8], sep: u128, quotes: u128) -> Option<[&[u8]; N]> {
    if sep.count_ones() as usize != N {
        return None;
    }
    let openings = (sep << 1) | 1;
    let closings = sep >> 1;
    // Each quote must be exclusively at the start or end of its field.
    if quotes & !(openings ^ closings) != 0 {
        return None;
    }
    // A carry from an opening quote runs through the field and stops at
    // its separator. The marked separators must match the closing quotes.
    let opened_ends = (!sep).wrapping_add(quotes & openings) & sep;
    if opened_ends != (quotes & closings) << 1 {
        return None;
    }
    if N == 7 {
        let low = sep as u64;
        let after_one = low & low.wrapping_sub(1);
        let after_two = after_one & after_one.wrapping_sub(1);
        let after_three = after_two & after_two.wrapping_sub(1);
        let high = (sep >> 64) as u64;
        if after_two != 0 && ((high >> 48) | (after_three & ((1 << 48) - 1))) == 0 {
            // The first three separators fit the first word; the final four
            // fit the overlapping 64-byte word beginning at byte 48.
            let mut suffix = (after_three >> 48) | (high << 16);
            let mut previous = 0;
            return Some(std::array::from_fn(|index| {
                let end = match index {
                    0 => low.trailing_zeros() as usize,
                    1 => after_one.trailing_zeros() as usize,
                    2 => after_two.trailing_zeros() as usize,
                    _ => {
                        let pos = 48 + suffix.trailing_zeros() as usize;
                        suffix &= suffix - 1;
                        pos
                    }
                };
                // The caller includes the terminating LF, so even the empty final field has a first byte.
                let trim = usize::from(unsafe { *line.get_unchecked(previous) } == b'"');
                // All offsets came from separators inside this row. Quoted
                // fields passed the paired and exclusive boundary checks.
                let slice = unsafe { line.get_unchecked(previous + trim..end - trim) };
                previous = end + 1;
                slice
            }));
        }
    }
    let mut sep_lo = sep as u64;
    let mut sep_hi = (sep >> 64) as u64;
    let mut previous = 0;
    let mut result = [&[][..]; N];
    for field in result.iter_mut() {
        let end = if sep_lo != 0 {
            let end = sep_lo.trailing_zeros() as usize;
            sep_lo &= sep_lo - 1;
            end
        } else {
            let end = 64 + sep_hi.trailing_zeros() as usize;
            sep_hi &= sep_hi - 1;
            end
        };
        let trim = usize::from(unsafe { *line.get_unchecked(previous) } == b'"');
        // Each separator is in the row and a trimmed field has two distinct quotes.
        *field = unsafe { line.get_unchecked(previous + trim..end - trim) };
        previous = end + 1;
    }
    Some(result)
}

impl<'a, const N: usize> Iterator for FastRows<'a, N> {
    type Item = [&'a [u8]; N];
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        while !self.data.is_empty() {
            let data = self.data;
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx2",
                target_feature = "bmi2"
            ))]
            if N < 32 && data.len() >= 128 {
                let a = classify(data[..64].try_into().unwrap());
                let (sep, quotes, len) = if a.newlines == 0 {
                    let b = classify_second(data[64..128].try_into().unwrap());
                    (
                        (a.separators as u128) | ((b.separators as u128) << 64),
                        (a.quotes as u128) | ((b.quotes as u128) << 64),
                        64 + b.newlines.trailing_zeros() as usize,
                    )
                } else {
                    (
                        a.separators as u128,
                        a.quotes as u128,
                        a.newlines.trailing_zeros() as usize,
                    )
                };
                if len < 128 {
                    let mask = u128::MAX >> (127 - len);
                    self.data = &data[len + 1..];
                    if let Some(row) = fast_record::<N>(&data[..=len], sep & mask, quotes & mask) {
                        return Some(row);
                    }
                    continue;
                }
            }
            let (line, next) = match memchr(b'\n', data) {
                Some(end) => (&data[..end], &data[end + 1..]),
                None => (data, &[][..]),
            };
            self.data = next;
            if let Some(row) = fields::<N>(line) {
                return Some(row);
            }
        }
        None
    }
}
// Split the short-ID prefix from the rest of the record at an actual field
// boundary. The remaining five fields fit in one machine word on typical
// rows, so their quote-pair validation never carries across a second word.
#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "bmi2"
    )
))]
#[derive(Clone, Copy)]
struct EventHead {
    mask: u32,
    commas: u32,
    quotes: u32,
    id_start: usize,
    user_start: usize,
    id_len: usize,
    user_len: usize,
    tail: usize,
}

#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "bmi2"
    )
))]
const fn event_heads() -> [EventHead; 16] {
    let mut result = [EventHead {
        mask: 0,
        commas: 0,
        quotes: 0,
        id_start: 0,
        user_start: 0,
        id_len: 0,
        user_len: 0,
        tail: 0,
    }; 16];
    let mut i = 0;
    while i < 16 {
        let id_width = 8 + (i & 3);
        let user_width = 8 + (i >> 2);
        let user_end = id_width + 1 + user_width;
        let id_quote = (i >> 1) & 1;
        let user_quote = (i >> 3) & 1;
        let quotes = (id_quote as u32) * (1 | (1 << (id_width - 1)))
            | (user_quote as u32) * ((1 << (id_width + 1)) | (1 << (user_end - 1)));
        result[i] = EventHead {
            mask: (1 << (user_end + 1)) - 1,
            commas: (1 << id_width) | (1 << user_end),
            quotes,
            id_start: id_quote,
            user_start: id_width + 1 + user_quote,
            id_len: id_width - 2 * id_quote,
            user_len: user_width - 2 * user_quote,
            tail: user_end + 1,
        };
        i += 1;
    }
    result
}
#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "bmi2"
    )
))]
const EVENT_HEADS: [EventHead; 16] = event_heads();

pub(crate) enum EventInput<'a> {
    Activity {
        suffix: u64,
        person: u64,
        timestamp: &'a [u8],
        arrival: &'a [u8],
    },
    General([&'a [u8]; 7]),
}

#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "bmi2"
    )
))]
#[inline(always)]
fn activity<'a>(
    line: &'a [u8],
    separators: u128,
    quotes: u128,
    prefix: u8,
) -> Option<EventInput<'a>> {
    // Store the optional quoting syntax in the prefix table. Its full mask
    // rejects an additional comma, an interior quote or a missing end quote.
    let first = (separators as u32).trailing_zeros() as usize;
    let second = ((separators as u32) & !0x1fff).trailing_zeros() as usize;
    let iw = first.wrapping_sub(8);
    let uw = second.wrapping_sub(first + 9);
    if (iw | uw) >= 4 {
        return None;
    }
    let head = &EVENT_HEADS[(uw << 2) | iw];
    if (((separators as u32) ^ head.commas) | ((quotes as u32) ^ head.quotes)) & head.mask != 0
        || line.len() <= head.tail
        || line.len() - head.tail > 64
        || line[head.id_start] != prefix
        || line[head.user_start] != b'u'
    {
        return None;
    }
    let tail = &line[head.tail..];
    let sep = (separators >> head.tail) as u64;
    let q = (quotes >> head.tail) as u64;
    let openings = (sep << 1) | 1;
    let closings = sep >> 1;
    if sep.count_ones() != 5
        || q & !(openings ^ closings) != 0
        || ((!sep).wrapping_add(q & openings) & sep) != ((q & closings) << 1)
    {
        return None;
    }
    let time_end = sep.trailing_zeros() as usize;
    let arrival_end = (sep & (sep - 1)).trailing_zeros() as usize;
    let kind_start = arrival_end + 1;
    let kind_q = ((q >> kind_start) & 1) as usize;
    let kind_data = kind_start + kind_q;
    let kind_end = kind_start + 4 + 2 * kind_q;
    if kind_end >= tail.len() || tail[kind_end] != b',' {
        return None;
    }
    let kind = u32::from_le_bytes(tail[kind_data..kind_data + 4].try_into().unwrap());
    if kind != u32::from_le_bytes(*b"view") && kind != u32::from_le_bytes(*b"cart") {
        return None;
    }
    // With four commas and an LF there is exactly one product delimiter
    // after a validated four-letter kind. A quoted empty product is invalid.
    let newline = tail.len() - 1;
    let product_end = 63 - (sep ^ (1u64 << newline)).leading_zeros() as usize;
    let product_start = kind_end + 1;
    let product_q = ((q >> product_start) & 1) as usize;
    if product_end <= product_start + 2 * product_q {
        return None;
    }
    let time_q = (q & 1) as usize;
    let arrival_start = time_end + 1;
    let arrival_q = ((q >> arrival_start) & 1) as usize;
    let timestamp = &tail[time_q..time_end - time_q];
    let arrival = &tail[arrival_start + arrival_q..arrival_end - arrival_q];
    if arrival.len() != 20 {
        return None;
    }
    let suffix = |at: usize, width: usize| {
        let word = u64::from_le_bytes(line[at + 1..at + 9].try_into().unwrap());
        word & (u64::MAX >> ((9 - width) * 8))
    };
    if line[head.id_start + head.id_len - 1] == 0 || line[head.user_start + head.user_len - 1] == 0
    {
        return None;
    }
    let suffix_id = suffix(head.id_start, head.id_len);
    let person = suffix(head.user_start, head.user_len);
    if person >> 63 != 0 {
        return None;
    }
    Some(EventInput::Activity {
        suffix: suffix_id,
        person,
        timestamp,
        arrival,
    })
}

#[inline(always)]
pub(crate) fn visit_event_rows<'a>(
    mut data: &'a [u8],
    _prefix: u8,
    mut emit: impl FnMut(EventInput<'a>),
) {
    while !data.is_empty() {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "bmi2"
        ))]
        if data.len() >= 128 {
            let a = classify(data[..64].try_into().unwrap());
            let (sep, quotes, len) = if a.newlines != 0 {
                (
                    a.separators as u128,
                    a.quotes as u128,
                    a.newlines.trailing_zeros() as usize,
                )
            } else {
                let b = classify_second(data[64..128].try_into().unwrap());
                (
                    u128::from(a.separators) | (u128::from(b.separators) << 64),
                    u128::from(a.quotes) | (u128::from(b.quotes) << 64),
                    64 + b.newlines.trailing_zeros() as usize,
                )
            };
            if len < 128 {
                let mask = u128::MAX >> (127 - len);
                let sep = sep & mask;
                let quotes = quotes & mask;
                let line = &data[..=len];
                data = &data[len + 1..];
                if let Some(fast) = activity(line, sep, quotes, _prefix) {
                    emit(fast);
                } else if let Some(row) = fast_record::<7>(line, sep, quotes) {
                    emit(EventInput::General(row));
                }
                continue;
            }
        }
        let (line, rest) = match memchr(b'\n', data) {
            Some(n) => (&data[..n], &data[n + 1..]),
            None => (data, &[][..]),
        };
        data = rest;
        if let Some(row) = fields::<7>(line) {
            emit(EventInput::General(row));
        }
    }
}

#[inline(always)]
fn eight_digits(digits: u64) -> i64 {
    let pairs = (digits.wrapping_mul(10) + (digits >> 8)) & 0x00ff_00ff_00ff_00ff;
    let quads = (pairs.wrapping_mul(100) + (pairs >> 16)) & 0x0000_ffff_0000_ffff;
    ((quads as u32 as i64) * 10_000) + (quads >> 32) as i64
}

pub fn unsigned(bytes: &[u8]) -> Option<i64> {
    if bytes.len() == 13 {
        const ASCII_ZERO: u64 = 0x3030_3030_3030_3030;
        // Split into five and eight digits with two in-bounds overlapping loads.
        let first = (u64::from_le_bytes(bytes[..8].try_into().unwrap()) ^ ASCII_ZERO) << 24;
        let last = u64::from_le_bytes(bytes[5..13].try_into().unwrap()) ^ ASCII_ZERO;
        if (first | last) & 0xf0f0_f0f0_f0f0_f0f0 != 0
            || (first.wrapping_add(0x0606_0606_0606_0606)
                | last.wrapping_add(0x0606_0606_0606_0606))
                & 0x1010_1010_1010_1010
                != 0
        {
            return None;
        }
        return Some(eight_digits(first) * 100_000_000 + eight_digits(last));
    }
    if bytes.is_empty() {
        return None;
    }
    if bytes.len() <= 18 {
        let (mut value, tail) = if let Some(first) = bytes.first_chunk::<8>() {
            let digits = u64::from_le_bytes(*first) ^ 0x3030_3030_3030_3030;
            if digits & 0xf0f0_f0f0_f0f0_f0f0 != 0
                || digits.wrapping_add(0x0606_0606_0606_0606) & 0x1010_1010_1010_1010 != 0
            {
                return None;
            }
            // Combine eight decimal digits in pairs, then as two 4-digit values.
            (eight_digits(digits), &bytes[8..])
        } else {
            (0, bytes)
        };
        for &byte in tail {
            let digit = byte.wrapping_sub(b'0');
            if digit > 9 {
                return None;
            }
            value = value * 10 + i64::from(digit);
        }
        return Some(value);
    }
    let mut value = 0i64;
    for &byte in bytes {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?.checked_add((byte - b'0') as i64)?;
    }
    Some(value)
}

fn digits(bytes: &[u8]) -> Option<i64> {
    let mut value = 0;
    for &byte in bytes {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value * 10 + (byte - b'0') as i64;
    }
    Some(value)
}

// Gregorian civil date to days since 1970-01-01, including leap centuries.
fn epoch_day(year: i64, month: i64, day: i64) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * month + 2) / 5 + day - 1;
    era * 146097 + year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year - 719468
}

pub fn timestamp(bytes: &[u8]) -> Option<i64> {
    TimestampParser::new().timestamp(bytes)
}

#[derive(Default, Clone, Copy)]
struct CachedDate {
    year_month: u64,
    day_bytes: u16,
    midnight: i64,
}

/// The same UTC dates occur repeatedly in a chunk. Cache only dates that
/// passed calendar validation, and compare all ten bytes on every hit.
#[derive(Default)]
pub struct TimestampParser {
    dates: [CachedDate; 32],
}

impl TimestampParser {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    fn midnight_seconds(&mut self, bytes: &[u8]) -> Option<i64> {
        let year_month = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let day_bytes = u16::from_le_bytes([bytes[8], bytes[9]]);
        // Modulo 32 this is 10 × the tens digit plus the units digit.
        let day = (u32::from(day_bytes) * 2561) >> 8;
        let slot = &mut self.dates[(day & 31) as usize];
        if slot.year_month != year_month || slot.day_bytes != day_bytes {
            if bytes[4] != b'-' || bytes[7] != b'-' {
                return None;
            }
            let year = digits(&bytes[..4])?;
            let month = digits(&bytes[5..7])?;
            let day = digits(&bytes[8..10])?;
            let last = match month {
                2 => 28 + i64::from(year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)),
                4 | 6 | 9 | 11 => 30,
                1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                _ => return None,
            };
            if year == 0 || day == 0 || day > last {
                return None;
            }
            *slot = CachedDate {
                year_month,
                day_bytes,
                midnight: epoch_day(year, month, day) * 86_400,
            };
        }
        Some(slot.midnight)
    }

    #[inline(always)]
    pub fn timestamp(&mut self, bytes: &[u8]) -> Option<i64> {
        if !matches!(bytes.len(), 20 | 25) || bytes[4] != b'-' {
            return unsigned(bytes);
        }
        if bytes[10] != b'T' {
            return None;
        }
        let word = u64::from_le_bytes(bytes[11..19].try_into().unwrap());
        let clock = word ^ u64::from_le_bytes(*b"00:00:00");
        // Require literal separators and six ASCII decimal digits. The high
        // nibble check catches non-ASCII bytes before decimal conversion.
        if clock & 0xf0f0_fff0_f0ff_f0f0 != 0
            || clock.wrapping_add(0x0606_0006_0600_0606) & 0x1010_0010_1000_1010 != 0
        {
            return None;
        }
        let seconds = clock_seconds::<true>(clock)?;
        let offset = if bytes.len() == 20 {
            if bytes[19] != b'Z' {
                return None;
            }
            0
        } else {
            if !matches!(bytes[19], b'+' | b'-') || bytes[22] != b':' {
                return None;
            }
            let hours = digits(&bytes[20..22])?;
            let minutes = digits(&bytes[23..25])?;
            if hours > 23 || minutes > 59 {
                return None;
            }
            (hours * 60 + minutes) * if bytes[19] == b'+' { 60 } else { -60 }
        };
        Some((self.midnight_seconds(bytes)? + seconds - offset) * 1000)
    }

    /// UTC event and arrival clocks often share a validated civil date.
    /// Compute both times of day in four SSE lanes and consult the date
    /// cache once. All other timestamp forms use the regular decoder.
    #[inline(always)]
    pub fn event_arrival(&mut self, event: &[u8], arrival: &[u8]) -> (i64, Option<i64>) {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "ssse3",
            target_feature = "sse4.1"
        ))]
        if event.len() == 20
            && arrival.len() == 20
            && event[10] == b'T'
            && event[19] == b'Z'
            && event[..10] == arrival[..10]
        {
            let event_clock = u64::from_le_bytes(event[11..19].try_into().unwrap())
                ^ u64::from_le_bytes(*b"00:00:00");
            // Validate both separators and all ASCII digits in the event.
            if event_clock & 0xf0f0_fff0_f0ff_f0f0 == 0
                && event_clock.wrapping_add(0x0606_0006_0600_0606) & 0x1010_0010_1000_1010 == 0
            {
                use std::arch::x86_64::*;
                let a =
                    u64::from_le_bytes(arrival[11..19].try_into().unwrap()) & 0x0f0f_000f_0f00_0f0f;
                let midnight = self
                    .midnight_seconds(arrival)
                    .expect("ingest_ts must have a calendar date");
                // Each 64-bit half contains HH, minute tens, minute units,
                // and seconds. The input contract guarantees an F1 arrival.
                unsafe {
                    let both = _mm_set_epi64x(a as i64, event_clock as i64);
                    let parts = _mm_maddubs_epi16(both, _mm_set1_epi64x(0x010a_0001_0a00_010a));
                    let invalid = _mm_cmpgt_epi16(parts, _mm_set1_epi64x(0x003b_0009_0032_0017));
                    let terms = _mm_madd_epi16(parts, _mm_set1_epi64x(0x0001_003c_003c_0e10));
                    let times = _mm_add_epi32(terms, _mm_srli_si128::<4>(terms));
                    let arrival_seconds = midnight + i64::from(_mm_extract_epi32::<2>(times));
                    let event_seconds = midnight + i64::from(_mm_cvtsi128_si32(times));
                    let valid = (_mm_cvtsi128_si64(invalid) == 0).then_some(event_seconds * 1000);
                    return (arrival_seconds, valid);
                }
            }
        }
        (self.ingest_seconds(arrival), self.timestamp(event))
    }

    /// The input contract guarantees that ingest_ts has valid UTC F1 syntax.
    #[inline(always)]
    pub fn ingest_seconds(&mut self, bytes: &[u8]) -> i64 {
        let bytes: &[u8; 20] = bytes.try_into().expect("ingest_ts must be valid F1");
        self.midnight_seconds(bytes)
            .expect("ingest_ts must have a calendar date")
            + clock_seconds::<false>(
                u64::from_le_bytes(bytes[11..19].try_into().unwrap()) & 0x0f0f_000f_0f00_0f0f,
            )
            .unwrap()
    }
}

#[inline(always)]
fn clock_seconds<const VALIDATE: bool>(clock: u64) -> Option<i64> {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "ssse3",
        target_feature = "sse4.1"
    ))]
    {
        use std::arch::x86_64::*;
        // Convert HH, the minute tens, the minute units and SS together.
        unsafe {
            let components = _mm_maddubs_epi16(
                _mm_cvtsi64_si128(clock as i64),
                _mm_cvtsi64_si128(0x010a_0001_0a00_010a),
            );
            if VALIDATE {
                let maxima = _mm_cvtsi64_si128(0x003b_0009_0032_0017);
                let bad = _mm_cmpgt_epi16(components, maxima);
                if _mm_testz_si128(bad, bad) == 0 {
                    return None;
                }
            }
            let weighted = _mm_madd_epi16(components, _mm_cvtsi64_si128(0x0001_003c_003c_0e10));
            let total = _mm_add_epi32(weighted, _mm_srli_si128::<4>(weighted));
            return Some(i64::from(_mm_cvtsi128_si32(total)));
        }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "ssse3",
        target_feature = "sse4.1"
    )))]
    {
        // Each two-digit decimal pair occupies its own 24-bit lane.
        const TENS: u64 = 0x000f_0000_0f00_000f;
        let pairs = (clock & TENS) * 10 + ((clock >> 8) & TENS);
        if VALIDATE && (pairs + 0x00c4_0000_c400_00e8) & 0x0100_0001_0000_0100 != 0 {
            return None;
        }
        let hour = (pairs as u8) as i64;
        let minute = ((pairs >> 24) as u8) as i64;
        let second = (pairs >> 48) as i64;
        Some(hour * 3600 + minute * 60 + second)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_dialect_keeps_spaces_and_carriage_returns() {
        assert_eq!(
            fields::<3>(b"\" u1\",\"\",1000\r"),
            Some([&b" u1"[..], &b""[..], &b"1000\r"[..]])
        );
        for row in [
            &b"u1,\"bad\"tail,p1"[..],
            b"u1,ba\"d,p1",
            b"u1,p1",
            b"u1,p1,2,3",
            b"",
        ] {
            assert!(fields::<3>(row).is_none());
        }
    }

    #[test]
    fn timestamps_are_strict_and_offsets_preserve_milliseconds() {
        assert_eq!(timestamp(b"1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(
            timestamp(b"2026-10-03T19:58:10-04:00"),
            timestamp(b"2026-10-03T23:58:10Z")
        );
        assert_eq!(timestamp(b"00000000000000086400000"), Some(DAY));
        assert_eq!(
            timestamp(b"2028-02-29T00:00:00-00:00"),
            timestamp(b"2028-02-29T00:00:00Z")
        );
        for value in [
            &b"2026-02-29T00:00:00Z"[..],
            b"2028-02-30T00:00:00Z",
            b"2028-01-01T24:00:00Z",
            b"2026-01-01T00:00:00+24:00",
            b"2026-01-01T00:00:00+05:60",
            b"2026-01-01T00:00:00z",
            b"1791071890000.0",
            b"99999999999999999999999999999999",
            b"2026-01-01T00:00:00Z\r",
        ] {
            assert!(timestamp(value).is_none(), "{:?}", value);
        }
    }

    #[test]
    fn block_csv_agrees_with_field_parser_at_all_alignments() {
        let cases: &[&[u8]] = &[
            b"a,b,c",
            b"aa,\"b\",c",
            b"\"\",,\"\"",
            b"\"\",,",
            b"aa,b\",c",
            b"aa,\",c",
            b",\"b\"z,c",
            b"\"a\"x,b,c",
            b"aa,b",
            b"aa,b,c,d",
            b"aa,b,c\r",
            b"\r,\" \",",
            b"a,,\r",
            b"\"\",a,b,c",
            b"",
        ];
        for prefix in 0..130 {
            let mut bytes = vec![b'x'; prefix];
            bytes.extend_from_slice(b",a,b\n");
            for &case in cases {
                bytes.extend_from_slice(case);
                bytes.push(b'\n');
            }
            bytes.extend_from_slice(b"\",");
            bytes.extend_from_slice(&[b'x'; 257]);
            bytes.extend_from_slice(b",y\n");
            bytes.extend_from_slice(&[b'z'; 257]);
            bytes.extend_from_slice(b",\":\",");
            for terminate in [false, true] {
                if terminate {
                    bytes.push(b'\n');
                }
                assert_eq!(
                    rows::<3>(&bytes).collect::<Vec<_>>(),
                    lines(&bytes).filter_map(fields::<3>).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn epoch_decimal_and_cached_calendar_boundaries() {
        let mut parser = TimestampParser::new();
        for n in [0, 1, 99999999, 1234567890123, 999999999999999999, i64::MAX] {
            for width in [1, 13, 20, 25, 100] {
                let text = format!("{n:0width$}");
                assert_eq!(parser.timestamp(text.as_bytes()), Some(n));
            }
        }
        assert_eq!(parser.ingest_seconds(b"2000-02-29T01:02:03Z"), 951786123);
        assert!(parser.timestamp(b"1900-02-29T01:02:03Z").is_none());
        assert_eq!(
            parser.timestamp(b"2000-02-29T01:02:03Z"),
            Some(951786123000)
        );
        for offset in [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18] {
            for bad in [0, b'/', b':', b'@', 0x80, 0xff] {
                let mut time = *b"2000-02-29T01:02:03Z";
                time[offset] = bad;
                assert_eq!(parser.timestamp(&time), None);
            }
        }
    }
}

#[cfg(test)]
mod fused_event_tests {
    use super::*;

    fn verify_projection(line: &[u8]) -> bool {
        assert!(line.len() <= 128 && line.last() == Some(&b'\n'));
        let mut separators = 0u128;
        let mut quotes = 0u128;
        for (i, &b) in line.iter().enumerate() {
            if b == b',' || b == b'\n' {
                separators |= 1u128 << i;
            }
            if b == b'"' {
                quotes |= 1u128 << i;
            }
        }
        let Some(EventInput::Activity {
            suffix,
            person,
            timestamp,
            arrival,
        }) = activity(line, separators, quotes, b'e')
        else {
            return false;
        };
        let [id, user, ts, ingest, kind, product, _] =
            fields::<7>(&line[..line.len() - 1]).unwrap();
        assert!(kind == b"view" || kind == b"cart");
        assert!(!product.is_empty());
        assert_eq!(timestamp, ts);
        assert_eq!(arrival, ingest);
        let word = |v: &[u8]| {
            let mut data = [0; 8];
            data[..v.len() - 1].copy_from_slice(&v[1..]);
            u64::from_le_bytes(data)
        };
        assert_eq!(id[0], b'e');
        assert_eq!(user[0], b'u');
        assert_ne!(id.last(), Some(&0));
        assert_ne!(user.last(), Some(&0));
        assert_eq!(suffix, word(id));
        assert_eq!(person, word(user));
        true
    }

    #[test]
    fn valid_activity_layouts_and_one_byte_edits_match_the_csv_parser() {
        for id in [&b"e0123456"[..], b"e01234567"] {
            for user in [&b"u7654321"[..], b"u87654321"] {
                for mask in 0..128 {
                    let mut line = Vec::new();
                    for (i, value) in [
                        id,
                        user,
                        b"1838243600000",
                        b"2028-04-01T00:01:22Z",
                        b"view",
                        b"p1",
                        b"",
                    ]
                    .iter()
                    .enumerate()
                    {
                        if i > 0 {
                            line.push(b',');
                        }
                        let quote = mask & (1 << i) != 0;
                        if quote {
                            line.push(b'"');
                        }
                        line.extend_from_slice(value);
                        if quote {
                            line.push(b'"');
                        }
                    }
                    line.push(b'\n');
                    assert!(verify_projection(&line));
                }
            }
        }
        for shape in [
            &b"e12345678,uabcdefgh,2028-04-01T00:00:00Z,2028-04-01T00:00:06Z,cart,p01,\n"[..],
            b"\"e1234567\",\"u12345678\",\"1838243600000\",\"2028-04-01T00:00:02Z\",\"view\",\"p1\",\"\"\n",
        ] {
            for i in 0..shape.len() - 1 {
                let mut line = shape.to_vec();
                for b in [0, b',', b'"', b' ', b'0', b'p', b'\r', 255] {
                    line[i] = b;
                    verify_projection(&line);
                }
            }
        }
    }

    #[test]
    fn whole_chunk_fast_and_fallback_owners_agree_before_validation() {
        let mut chunk = Vec::new();
        for line in [
            &b"e12345678,u12345678,invalid,2028-02-29T05:00:00Z,view,p2,"[..],
            b"\"e12345678\",\"u76543210\",2028-02-29T04:00:00Z,2028-02-29T05:00:01Z,\"cart\",p2,",
            b"eabc45678,u12345678,1838243600000,2028-02-29T05:00:00Z,purchase,p2,10",
            b"e12345678,u12345678,1838243600000,2028-02-29T04:00:00Z,view,p2,broken\"quote",
            b"arbitrary full width event id,u\0x,2028-02-29T02:00:00-03:00,2028-02-29T05:00:00Z,view,p1,",
            b"e1234567,u12345678,2028-02-29T03:00:00Z,2028-02-29T04:00:00Z,view,\"\",",
        ] {
            chunk.extend_from_slice(line);
            chunk.push(b'\n');
        }
        chunk.extend(std::iter::repeat_n(b'x', 150));
        chunk.extend_from_slice(b",u12345678,1700000000000,2028-02-29T05:00:00Z,cart,p1,");
        let general = |row: [&[u8]; 7]| {
            let [id, user, time, arrival, kind, product, price] = row;
            let mut dates = TimestampParser::new();
            let seconds = dates.ingest_seconds(arrival);
            let valid = !user.is_empty()
                && !product.is_empty()
                && (kind == b"view"
                    || kind == b"cart"
                    || (kind == b"purchase"
                        && unsigned(price).is_some_and(|n| (1..=10_000_000).contains(&n))));
            let payload = dates
                .timestamp(time)
                .filter(|&t| valid && seconds * 1000 - t <= DAY)
                .map(|t| (user.to_vec(), t));
            (id.to_vec(), seconds, payload)
        };
        let expected: Vec<_> = lines(&chunk).filter_map(fields::<7>).map(general).collect();
        let mut actual = Vec::new();
        visit_event_rows(&chunk, b'e', |row| {
            actual.push(match row {
                EventInput::General(row) => general(row),
                EventInput::Activity {
                    suffix,
                    person,
                    timestamp,
                    arrival,
                } => {
                    let name = |first, word: u64| {
                        let bytes = word.to_le_bytes();
                        let end = 8 - word.leading_zeros() as usize / 8;
                        let mut value = vec![first];
                        value.extend_from_slice(&bytes[..end]);
                        value
                    };
                    let (seconds, time) = TimestampParser::new().event_arrival(timestamp, arrival);
                    (
                        name(b'e', suffix),
                        seconds,
                        time.filter(|&t| seconds * 1000 - t <= DAY)
                            .map(|t| (name(b'u', person), t)),
                    )
                }
            })
        });
        assert_eq!(actual, expected);
        let winner = actual
            .iter()
            .filter(|(id, _, _)| id == b"e12345678")
            .min_by_key(|row| row.1)
            .unwrap();
        assert!(winner.2.is_none());
    }

    #[test]
    fn paired_timestamps_match_independent_decoding_at_clock_and_date_edges() {
        let mut parser = TimestampParser::new();
        for raw in [
            &b"2028-02-29T23:59:59Z"[..],
            b"2028-02-29T24:00:00Z",
            b"2028-02-29T00:60:00Z",
            b"2028-02-29T00:00:60Z",
            b"2028-02-28T22:05:00-04:30",
            b"2028-02-29T11:30:40+08:45",
            b"1845756335025",
            b"0000000",
            b"2027-02-29T12:00:00Z",
            b"bad timestamp",
            b"",
        ] {
            let arrival = b"2028-02-29T23:17:09Z";
            assert_eq!(
                parser.event_arrival(raw, arrival),
                (parser.ingest_seconds(arrival), parser.timestamp(raw))
            );
        }
        for index in 10..20 {
            let arrival = b"2028-02-29T01:02:03Z";
            let mut event = *b"2028-02-29T11:22:33Z";
            for byte in 0u8..=255 {
                event[index] = byte;
                assert_eq!(
                    parser.event_arrival(&event, arrival),
                    (parser.ingest_seconds(arrival), parser.timestamp(&event))
                );
            }
        }
    }
}
