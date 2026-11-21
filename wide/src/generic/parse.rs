use super::util::{count_byte, csv_mask64, eq_mask64, find_byte, find_sub};
use std::io;

pub type B<'a> = &'a [u8];

pub const DAY_MS: i64 = 86_400_000;

pub fn chunks<'a>(files: &[B<'a>], target: usize) -> Vec<B<'a>> {
    let target = target.max(1);
    let mut out = Vec::new();
    for &f in files {
        let body = match find_byte(f, b'\n') {
            Some(p) => &f[p + 1..],
            None => continue,
        };
        let mut s = 0;
        while s < body.len() {
            let mut e = s.saturating_add(target).min(body.len());
            if e < body.len() {
                e = match find_byte(&body[e..], b'\n') {
                    Some(p) => e + p + 1,
                    None => body.len(),
                };
            }
            out.push(&body[s..e]);
            s = e;
        }
    }
    out
}

pub fn chunk_target(files: &[B], ncpu: usize) -> usize {
    let total: usize = files.iter().map(|f| f.len()).sum();
    (total / (ncpu * 16).max(1)).clamp(1 << 20, 16 << 20)
}

#[inline]
pub fn count_lines(c: B) -> usize {
    count_byte(c, b'\n')
}

#[inline]
pub fn for_lines<'a>(chunk: B<'a>, mut f: impl FnMut(B<'a>)) {
    let n = chunk.len();
    let mut s = 0usize;
    let mut base = 0usize;
    while base + 64 <= n {
        let mut m = eq_mask64(&chunk[base..], b'\n');
        while m != 0 {
            let p = base + m.trailing_zeros() as usize;
            if p > s {
                f(&chunk[s..p]);
            }
            s = p + 1;
            m &= m - 1;
        }
        base += 64;
    }
    for p in base..n {
        if chunk[p] == b'\n' {
            if p > s {
                f(&chunk[s..p]);
            }
            s = p + 1;
        }
    }
    if s < n {
        f(&chunk[s..]);
    }
}

#[inline]
fn unquote(f: B, has_quote: bool) -> Option<B> {
    if !has_quote {
        return Some(f);
    }
    let n = f.len();
    if n >= 2 && f[0] == b'"' && f[n - 1] == b'"' && !f[1..n - 1].contains(&b'"') {
        Some(&f[1..n - 1])
    } else {
        None
    }
}

#[inline]
pub fn split<'a, const N: usize>(line: B<'a>) -> Option<[B<'a>; N]> {
    let mut out: [B<'a>; N] = [&[]; N];
    let mut k = 0usize;
    let mut start = 0usize;
    let mut q = false;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if k + 1 >= N {
                return None;
            }
            out[k] = unquote(&line[start..i], q)?;
            k += 1;
            start = i + 1;
            q = false;
        } else if b == b'"' {
            q = true;
        }
    }
    if k + 1 != N {
        return None;
    }
    out[k] = unquote(&line[start..], q)?;
    Some(out)
}

/// Tokenize a whole chunk once. Field boundaries survive SIMD block boundaries.
#[inline]
pub fn for_records<'a, const N: usize>(chunk: B<'a>, mut row: impl FnMut([B<'a>; N])) {
    let mut fields: [B<'a>; N] = [&[]; N];
    let mut line = 0;
    let mut start = 0;
    let mut column = 0;
    let mut quotes = 0usize;
    let mut marker = |end: usize, byte: u8| {
        if byte == b'"' {
            quotes += 1;
            return;
        }
        if column < N {
            let field = &chunk[start..end];
            if quotes == 0 {
                fields[column] = field;
                column += 1;
            } else if quotes == 2 && field.len() >= 2 && field[0] == b'"' && field[field.len() - 1] == b'"' {
                fields[column] = &field[1..field.len() - 1];
                column += 1;
            } else {
                column = N + 1;
            }
        } else {
            column = N + 1;
        }
        if byte == b'\n' {
            if column == N && end > line {
                row(fields);
            }
            line = end + 1;
            column = 0;
        }
        quotes = 0;
        start = end + 1;
    };
    let mut base = 0;
    while chunk.len() - base >= 64 {
        let mut mask = csv_mask64(&chunk[base..]);
        while mask != 0 {
            let pos = base + mask.trailing_zeros() as usize;
            marker(pos, chunk[pos]);
            mask &= mask - 1;
        }
        base += 64;
    }
    for (pos, &b) in chunk.iter().enumerate().skip(base) {
        if matches!(b, b',' | b'"' | b'\n') {
            marker(pos, b);
        }
    }
    if !chunk.is_empty() && chunk.last() != Some(&b'\n') {
        marker(chunk.len(), b'\n');
    }
}

#[inline]
fn d2(a: u8, b: u8) -> Option<i64> {
    if a.is_ascii_digit() && b.is_ascii_digit() {
        Some(((a - b'0') * 10 + (b - b'0')) as i64)
    } else {
        None
    }
}

pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn is_leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

#[inline]
fn midnight(s: &[u8]) -> Option<i64> {
    if s[4] != b'-' || s[7] != b'-' {
        return None;
    }
    let y = d2(s[0], s[1])? * 100 + d2(s[2], s[3])?;
    let month = d2(s[5], s[6])?;
    let day = d2(s[8], s[9])?;
    let limit = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(y) => 29,
        2 => 28,
        _ => return None,
    };
    if day == 0 || day > limit {
        return None;
    }
    Some(days_from_civil(y, month, day) * DAY_MS)
}

#[derive(Default, Clone, Copy)]
struct DateSlot {
    prefix: u64,
    day: u16,
    midnight: i64,
}

pub struct TimestampParser {
    days: [DateSlot; 64],
}

impl TimestampParser {
    pub fn new() -> Self {
        Self {
            days: [DateSlot::default(); 64],
        }
    }

    #[inline]
    pub fn parse(&mut self, s: &[u8]) -> Option<i64> {
        if matches!(s.len(), 20 | 25) && s[4] == b'-' {
            let offset = match s.len() {
                20 if s[19] == b'Z' => 0,
                25 if s[22] == b':' => {
                    let hours = d2(s[20], s[21])?;
                    let mins = d2(s[23], s[24])?;
                    if hours > 23 || mins > 59 {
                        return None;
                    }
                    let offset = (hours * 60 + mins) * 60_000;
                    match s[19] {
                        b'+' => offset,
                        b'-' => -offset,
                        _ => return None,
                    }
                }
                _ => return None,
            };
            if s[10] != b'T' || s[13] != b':' || s[16] != b':' {
                return None;
            }
            let h = d2(s[11], s[12])?;
            let m = d2(s[14], s[15])?;
            let sec = d2(s[17], s[18])?;
            if h > 23 || m > 59 || sec > 59 {
                return None;
            }
            // Use the day digits to select a slot. Exact equality on all ten bytes
            // makes this independent of the window, data order and date range.
            let p = u64::from_le_bytes(s[..8].try_into().unwrap());
            let day = u16::from_le_bytes(s[8..10].try_into().unwrap());
            let idx = ((s[8] as usize * 11) + s[9] as usize) & 63;
            let slot = &mut self.days[idx];
            if slot.prefix != p || slot.day != day {
                *slot = DateSlot {
                    prefix: p,
                    day,
                    midnight: midnight(s)?,
                };
            }
            Some(slot.midnight + ((h * 60 + m) * 60 + sec) * 1000 - offset)
        } else {
            // Values beyond the supported civil window can be dropped straight away.
            // Skip leading zeros once so arbitrarily padded epoch values still work.
            if s.is_empty() {
                return None;
            }
            let digits = &s[s.iter().position(|&b| b != b'0').unwrap_or(s.len())..];
            if digits.len() > 18 {
                return None;
            }
            let mut value = 0i64;
            for &b in digits {
                let d = b.wrapping_sub(b'0');
                if d > 9 {
                    return None;
                }
                value = value * 10 + i64::from(d);
            }
            Some(value)
        }
    }
}

pub fn parse_ts(s: &[u8]) -> Option<i64> {
    TimestampParser::new().parse(s)
}

#[inline]
pub fn parse_price(s: &[u8]) -> Option<u32> {
    if s.is_empty() || !s.iter().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut i = 0;
    while i < s.len() && s[i] == b'0' {
        i += 1;
    }
    let r = &s[i..];
    if r.len() > 8 {
        return None;
    }
    let mut v: u32 = 0;
    for &b in r {
        v = v * 10 + (b - b'0') as u32;
    }
    if (1..=10_000_000).contains(&v) {
        Some(v)
    } else {
        None
    }
}

#[inline]
pub fn day_of(ts: i64) -> i32 {
    ts.div_euclid(DAY_MS) as i32
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

pub fn manifest(data: &[u8]) -> io::Result<(i64, i64)> {
    let get = |key: &[u8]| -> io::Result<i64> {
        let k = String::from_utf8_lossy(key).into_owned();
        let pos = find_sub(data, key).ok_or_else(|| bad(format!("manifest: no {k}")))?;
        let rest = &data[pos + key.len()..];
        let colon = find_byte(rest, b':').ok_or_else(|| bad(format!("manifest: no colon after {k}")))?;
        let rest = &rest[colon + 1..];
        let q1 = find_byte(rest, b'"').ok_or_else(|| bad(format!("manifest: no value for {k}")))?;
        let rest = &rest[q1 + 1..];
        let q2 = find_byte(rest, b'"').ok_or_else(|| bad(format!("manifest: unterminated {k}")))?;
        parse_ts(&rest[..q2]).ok_or_else(|| bad(format!("manifest: bad timestamp {}", String::from_utf8_lossy(&rest[..q2]))))
    };
    Ok((get(b"\"window_start\"")?, get(b"\"window_end\"")?))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ts() {
        assert_eq!(parse_ts(b"2026-10-03T23:58:10Z"), Some(1791071890000));
        assert_eq!(parse_ts(b"2026-10-03T19:58:10-04:00"), Some(1791071890000));
        assert_eq!(parse_ts(b"1791071890000"), Some(1791071890000));
        assert_eq!(parse_ts(b"0001791071890000"), Some(1791071890000));
        assert_eq!(parse_ts(b"2026-10-03T23:58:10z"), None);
        assert_eq!(parse_ts(b"2026-02-29T00:00:00Z"), None);
        assert_eq!(parse_ts(b"2028-02-29T00:00:00Z").is_some(), true);
        assert_eq!(parse_ts(b"2026-10-03T19:58:10+24:00"), None);
        assert_eq!(parse_ts(b"2026-10-03T19:58:10+0200"), None);
        assert_eq!(parse_ts(b"2026-10-03T19:58:10-00:00"), parse_ts(b"2026-10-03T19:58:10Z"));
        assert_eq!(parse_ts(b"2026-10-03T24:00:00Z"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:59:60Z"), None);
        assert_eq!(parse_ts(b"1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_price(b"0250"), Some(250));
        assert_eq!(parse_price(b"10000000"), Some(10_000_000));
        assert_eq!(parse_price(b"10000001"), None);
        assert_eq!(parse_price(b"0"), None);
        assert_eq!(split::<2>(b"\"a\",\"\""), Some([&b"a"[..], &b""[..]]));
        assert_eq!(split::<2>(b"a,b\""), None);
        assert_eq!(split::<2>(b"a,b,c"), None);
        assert_eq!(split::<2>(b"a"), None);
    }
    #[test]
    fn structural_csv_matches_per_line_parser() {
        // Shift all delimiter and quote positions through every SIMD lane.
        // The final record has no trailing line feed.
        let rows = b"a,b,c\n\"left\",\"\",right\n\na,b\na,b,c,d\n\"a,b,c\na\"b,c,d\n\"a\"\"\",b,c\na\r,b,c\r\n one,\" two\",last";
        for n in 0..128 {
            let mut bytes = vec![b'x'; n];
            bytes.extend_from_slice(rows);
            let mut expected = Vec::new();
            for_lines(&bytes, |line| {
                if let Some(fields) = split::<3>(line) {
                    expected.push(fields);
                }
            });
            let mut actual = Vec::new();
            for_records::<3>(&bytes, |fields| actual.push(fields));
            assert_eq!(actual, expected, "alignment {n}");
        }
    }

    #[test]
    fn dates_with_same_cache_index_are_verified() {
        let mut parser = TimestampParser::new();
        for (text, answer) in [
            (&b"2028-02-29T00:00:00Z"[..], Some(1_835_395_200_000)),
            (b"2027-02-29T00:00:00Z", None),
            (b"2028-02-29T00:00:00-01:30", Some(1_835_400_600_000)),
            (b"2028-00-29T00:00:00Z", None),
            (b"00000000000000000000000000000000000000000000001", Some(1)),
            (b"0000000000000000000000000000000000000000000000x", None),
        ] {
            assert_eq!(parser.parse(text), answer);
        }
    }

    #[test]
    fn lines() {
        let mut s = Vec::new();
        for i in 0..300 {
            s.extend_from_slice(format!("line{i}\n").as_bytes());
            if i % 7 == 0 {
                s.extend_from_slice(b"\n\n");
            }
        }
        s.extend_from_slice(b"tail");
        let mut got = Vec::new();
        for_lines(&s, |l| got.push(l.to_vec()));
        assert_eq!(got.len(), 301);
        assert_eq!(got[300], b"tail".to_vec());
        assert_eq!(count_lines(&s), 300 + 2 * 43);
        assert_eq!(find_byte(&s[1..], b'\n'), Some(4));
    }
}
