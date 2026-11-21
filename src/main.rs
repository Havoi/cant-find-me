mod ingest;
mod input;
mod output;
mod parse;

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use ahash::{AHashMap, RandomState};
use rayon::prelude::*;

use input::{Dimensions, Inputs, People};
use output::UserDay;
use parse::{timestamp, DAY, SESSION_GAP, WEEK};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
type CampaignTotals = AHashMap<(u32, i32), CampaignStats>;

// Use separate bits of the same uniformly mixed word for owner partitions,
// person partitions and the lookup table inside each person partition.
#[inline]
fn mix_key(key: u64) -> u64 {
    (key ^ (key >> 29)).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

#[derive(Clone, Copy)]
struct PersonEvent {
    person: u64,
    ts: i64,
    revenue: u32,
    category: u32,
}

trait ClickTime: Copy + Default + Ord + Send + Sync {
    fn encode(timestamp: i64, origin: i64) -> Self;
}

impl ClickTime for u32 {
    #[inline]
    fn encode(timestamp: i64, origin: i64) -> Self {
        // The compact path checks the entire inclusive click range once.
        (timestamp - origin) as u32
    }
}

impl ClickTime for i64 {
    #[inline]
    fn encode(timestamp: i64, _: i64) -> Self {
        timestamp
    }
}

#[derive(Clone, Copy, Default)]
struct Ad<C> {
    id: u64,
    time: C,
    campaign: u32,
}

struct KeyedClick<C> {
    person: u64,
    click: Ad<C>,
}

#[derive(Clone, Copy, Default)]
struct CampaignStats {
    sessions: i64,
    purchasing_sessions: i64,
    revenue: i64,
    assists: i64,
}

impl CampaignStats {
    fn add(&mut self, other: Self) {
        self.sessions += other.sessions;
        self.purchasing_sessions += other.purchasing_sessions;
        self.revenue += other.revenue;
        self.assists += other.assists;
    }
}

// A short manifest window permits direct indexing for both credit and assists.
// Keep a sparse representation when the date range or catalog is large.
enum DailyCampaigns {
    Dense {
        first_day: i32,
        width: usize,
        cells: Vec<CampaignStats>,
    },
    Sparse(CampaignTotals),
}

impl DailyCampaigns {
    fn new(context: &Context) -> Self {
        let first_day = context.start.div_euclid(DAY);
        let days = (context.end - 1).div_euclid(DAY) - first_day + 1;
        let width = context.dimensions.campaign_names.len();
        match (days as usize).checked_mul(width) {
            Some(size) if size <= 131072 => Self::Dense {
                first_day: first_day as i32,
                width,
                cells: vec![CampaignStats::default(); size],
            },
            _ => Self::Sparse(CampaignTotals::new()),
        }
    }

    fn record(&mut self, day: i32, campaign: u32, revenue: i64, assists: &[u32]) {
        let credited = match self {
            Self::Dense {
                first_day,
                width,
                cells,
            } => {
                let begin = (day - *first_day) as usize * *width;
                let daily = &mut cells[begin..begin + *width];
                for &assist in assists {
                    daily[assist as usize].assists += 1;
                }
                &mut daily[campaign as usize]
            }
            Self::Sparse(totals) => {
                for &assist in assists {
                    totals.entry((assist, day)).or_default().assists += 1;
                }
                totals.entry((campaign, day)).or_default()
            }
        };
        credited.sessions += 1;
        credited.purchasing_sessions += i64::from(revenue > 0);
        credited.revenue += revenue;
    }

    fn merge(mut self, other: Self) -> Self {
        match (&mut self, other) {
            (Self::Dense { cells: left, .. }, Self::Dense { cells: right, .. }) => {
                for (a, b) in left.iter_mut().zip(right) {
                    a.add(b);
                }
            }
            (Self::Sparse(left), Self::Sparse(right)) => {
                for (key, stats) in right {
                    left.entry(key).or_default().add(stats);
                }
            }
            _ => unreachable!("all shards share one manifest and catalog"),
        }
        self
    }

    fn into_totals(self) -> CampaignTotals {
        match self {
            Self::Dense {
                first_day,
                width,
                cells,
            } => cells
                .chunks(width)
                .enumerate()
                .flat_map(|(offset, day)| {
                    day.iter()
                        .copied()
                        .enumerate()
                        .filter(|(_, c)| c.sessions != 0 || c.assists != 0)
                        .map(move |(campaign, c)| ((campaign as u32, first_day + offset as i32), c))
                })
                .collect(),
            Self::Sparse(totals) => totals,
        }
    }
}

struct Context<'a> {
    click_sources: Vec<&'a [u8]>,
    people: People,
    dimensions: Dimensions,
    start: i64,
    end: i64,
    hash: RandomState,
    person_shards: usize,
    event_shards: usize,
    output: PathBuf,
}

impl Context<'_> {
    fn click_origin(&self) -> i64 {
        self.start.saturating_sub(WEEK)
    }

    // An ID is the first field of a validated click row. The parser has
    // already stripped its opening quote, and the following comma or quote
    // always terminates the value, even when it contains NUL or UTF-8 bytes.
    fn click_id(&self, id: u64) -> &[u8] {
        let data = &self.click_sources[(id >> 32) as usize][id as u32 as usize..];
        &data[..memchr::memchr2(b',', b'"', data).expect("a validated click has four fields")]
    }
    fn person(&self, user: &[u8]) -> u64 {
        self.people.resolve(user)
    }

    fn person_shard(&self, person: u64) -> usize {
        ((mix_key(person) >> 32) as usize) & (self.person_shards - 1)
    }
}

fn buckets<T>(count: usize) -> Vec<Vec<T>> {
    (0..count).map(|_| Vec::new()).collect()
}

// Transpose ownership of buffers without copying their rows.
fn transpose<T>(groups: Vec<Vec<Vec<T>>>, count: usize) -> Vec<Vec<Vec<T>>> {
    let mut result = buckets(count);
    for group in groups {
        for (index, values) in group.into_iter().enumerate() {
            if !values.is_empty() {
                result[index].push(values);
            }
        }
    }
    result
}

struct Attribution {
    counts: Vec<usize>,
    positions: Vec<usize>,
    active: Vec<u32>,
    head: usize,
    tail: usize,
    winner: Option<usize>,
}

impl Attribution {
    fn new(campaigns: usize) -> Self {
        Self {
            counts: vec![0; campaigns],
            positions: vec![0; campaigns],
            active: Vec::new(),
            head: 0,
            tail: 0,
            winner: None,
        }
    }

    fn reset(&mut self) {
        for campaign in self.active.drain(..) {
            self.counts[campaign as usize] = 0;
        }
        self.head = 0;
        self.tail = 0;
        self.winner = None;
    }

    fn advance<C: ClickTime>(&mut self, clicks: &[Ad<C>], start: i64, context: &Context) -> u32 {
        let origin = context.click_origin();
        let latest = C::encode(start, origin);
        let earliest = C::encode(start.saturating_sub(WEEK), origin);
        while self.tail < clicks.len() && clicks[self.tail].time <= latest {
            let click = &clicks[self.tail];
            let campaign = click.campaign as usize;
            if self.counts[campaign] == 0 {
                self.positions[campaign] = self.active.len();
                self.active.push(click.campaign);
            }
            self.counts[campaign] += 1;
            // Only timestamp ties require looking up the original ID bytes.
            if self.winner.is_none_or(|winner| {
                let old = &clicks[winner];
                old.time < click.time
                    || (old.time == click.time
                        && context.click_id(click.id) < context.click_id(old.id))
            }) {
                self.winner = Some(self.tail);
            }
            self.tail += 1;
        }
        while self.head < self.tail && clicks[self.head].time < earliest {
            let campaign = clicks[self.head].campaign as usize;
            self.counts[campaign] -= 1;
            if self.counts[campaign] == 0 {
                let position = self.positions[campaign];
                self.active.swap_remove(position);
                if position < self.active.len() {
                    self.positions[self.active[position] as usize] = position;
                }
            }
            self.head += 1;
        }
        self.winner
            .filter(|&winner| winner >= self.head)
            .map_or(context.dimensions.organic, |winner| clicks[winner].campaign)
    }
}

fn finish_day(rows: &mut Vec<UserDay>, mut row: UserDay, categories: &mut [i64]) {
    if row.revenue == 0 {
        rows.push(row);
        return;
    }
    let mut highest = 0;
    // Category indices are assigned in byte order, so the first maximum wins.
    for (index, revenue) in categories.iter_mut().enumerate() {
        if *revenue > highest {
            highest = *revenue;
            row.category = Some(index as u32);
        }
        *revenue = 0;
    }
    rows.push(row);
}

// Event timestamps are stored separately from person identity after partitioning.
// A short window fits an exact millisecond timestamp and a purchase index in
// one sortable word. All other manifests retain full-width event fields.
struct KeyedEvent<T> {
    person: u64,
    data: T,
}
trait GroupEvent: Copy + Default + Ord + Send + Sync {
    fn from_packed(offset: u32, finance: u32, start: i64) -> Self {
        Self::pack(
            PersonEvent {
                person: 0,
                ts: start + i64::from(offset),
                revenue: finance & 0xff_ffff,
                category: finance >> 24,
            },
            start,
        )
    }
    fn pack(row: PersonEvent, start: i64) -> Self;
    fn time(self, start: i64) -> i64;
    fn purchase(self) -> Option<(u32, u32)>;
}

impl GroupEvent for u64 {
    #[inline]
    fn from_packed(offset: u32, finance: u32, _: i64) -> Self {
        (u64::from(offset) << 32) | u64::from(finance)
    }
    #[inline]
    fn pack(row: PersonEvent, start: i64) -> Self {
        // Prices pass ingestion's inclusive 10,000,000-cent limit (24 bits).
        ((row.ts - start) as u64) << 32 | (u64::from(row.category) << 24) | u64::from(row.revenue)
    }
    #[inline]
    fn time(self, start: i64) -> i64 {
        start + (self >> 32) as i64
    }
    #[inline]
    fn purchase(self) -> Option<(u32, u32)> {
        let price = (self as u32) & 0xff_ffff;
        (price != 0).then_some((price, (self as u32) >> 24))
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
struct WideEvent {
    ts: i64,
    revenue: u32,
    category: u32,
}

impl GroupEvent for WideEvent {
    fn pack(row: PersonEvent, _: i64) -> Self {
        Self {
            ts: row.ts,
            revenue: row.revenue,
            category: row.category,
        }
    }
    fn time(self, _: i64) -> i64 {
        self.ts
    }
    fn purchase(self) -> Option<(u32, u32)> {
        (self.revenue != 0).then_some((self.revenue, self.category))
    }
}

#[derive(Clone, Copy, Default)]
struct PersonRange {
    person: u64,
    event_end: usize,
    click_end: usize,
}

#[derive(Clone, Copy)]
struct TimeBin {
    first: i64,
    last: i64,
    events: i64,
    revenue: i64,
}

#[derive(Default)]
struct BinScratch {
    bins: Vec<TimeBin>,
    revenue: Vec<i64>,
}

impl BinScratch {
    fn record<T: GroupEvent, C: ClickTime>(
        &mut self,
        person: u64,
        events: &[T],
        clicks: &[Ad<C>],
        context: &Context,
        bin_count: usize,
        attribution: &mut Attribution,
        totals: &mut DailyCampaigns,
        rows: &mut Vec<UserDay>,
        categories: &mut [i64],
    ) {
        // The width equals the allowed inactivity gap. All events in a bin
        // necessarily belong to one session. For two adjacent occupied bins
        // only the previous maximum and the next minimum can separate them.
        let empty = TimeBin {
            first: i64::MAX,
            last: i64::MIN,
            events: 0,
            revenue: 0,
        };
        self.bins.clear();
        self.bins.resize(bin_count, empty);
        self.revenue.clear();
        self.revenue.resize(bin_count * categories.len(), 0);
        for event in events {
            let ts = event.time(context.start);
            let index = ((ts - context.start) / SESSION_GAP) as usize;
            let bin = &mut self.bins[index];
            bin.first = bin.first.min(ts);
            bin.last = bin.last.max(ts);
            bin.events += 1;
            if let Some((revenue, category)) = event.purchase() {
                self.revenue[index * categories.len() + category as usize] += i64::from(revenue);
                bin.revenue += i64::from(revenue);
            }
        }
        let mut current: Option<UserDay> = None;
        let mut at = 0;
        while at < self.bins.len() {
            let first = self.bins[at];
            if first.events == 0 {
                at += 1;
                continue;
            }
            let day = first.first.div_euclid(DAY) as i32;
            if current.as_ref().is_some_and(|row| row.day != day) {
                finish_day(rows, current.take().unwrap(), categories);
            }
            let row = current.get_or_insert(UserDay {
                person,
                day,
                sessions: 0,
                events: 0,
                revenue: 0,
                category: None,
            });
            let mut last = first.first;
            let mut revenue = 0;
            while at < self.bins.len() {
                let bin = self.bins[at];
                if bin.events != 0 {
                    if bin.first - last > SESSION_GAP {
                        break;
                    }
                    row.events += bin.events;
                    revenue += bin.revenue;
                    if bin.revenue != 0 {
                        let begin = at * categories.len();
                        for (category, subtotal) in
                            categories.iter_mut().zip(&self.revenue[begin..])
                        {
                            *category += subtotal;
                        }
                    }
                    last = bin.last;
                }
                at += 1;
            }
            row.sessions += 1;
            row.revenue += revenue;
            let campaign = attribution.advance(clicks, first.first, context);
            totals.record(day, campaign, revenue, &attribution.active);
        }
        if let Some(row) = current {
            finish_day(rows, row, categories);
        }
    }
}

/// A worker owns one complete set of buffers while it drains person shards.
/// Every grouped event and click slot is overwritten before being read.
struct PersonWorker<T, C> {
    grouped_events: Vec<T>,
    grouped_clicks: Vec<Ad<C>>,
    ranges: Vec<PersonRange>,
    overflow: AHashMap<u64, usize>,
    rows: Vec<UserDay>,
    categories: Vec<i64>,
    attribution: Attribution,
    bins: BinScratch,
    totals: DailyCampaigns,
}

impl<T, C> PersonWorker<T, C> {
    fn new(context: &Context) -> Self {
        Self {
            grouped_events: Vec::new(),
            grouped_clicks: Vec::new(),
            ranges: Vec::new(),
            overflow: AHashMap::new(),
            rows: Vec::new(),
            categories: vec![0; context.dimensions.categories.len()],
            attribution: Attribution::new(context.dimensions.campaign_names.len()),
            bins: BinScratch::default(),
            totals: DailyCampaigns::new(context),
        }
    }
}

fn group_people<T: GroupEvent, C: ClickTime>(
    mut events: Vec<Vec<KeyedEvent<T>>>,
    mut clicks: Vec<Vec<KeyedClick<C>>>,
    shard: usize,
    context: &Context,
    count: usize,
    worker: &mut PersonWorker<T, C>,
) -> Result<()> {
    let PersonWorker {
        grouped_events,
        grouped_clicks,
        ranges,
        overflow,
        rows,
        categories,
        attribution,
        bins,
        totals,
    } = worker;
    // The usual case needs only one cache-line probe for both identity and
    // counts. A separate overflow map keeps the probe bound fixed even for a
    // shard in which every event has a different person.
    let slots = (count / 16).max(16).next_power_of_two();
    let mask = slots - 1;
    let shift = 64 - slots.trailing_zeros();
    let first_slot = |person: u64| {
        let bits = mix_key(person);
        (bits >> shift) as usize
    };
    ranges.clear();
    ranges.resize(slots, PersonRange::default());
    overflow.clear();
    for row in events.iter_mut().flatten() {
        let first = first_slot(row.person);
        let mut found = None;
        for probe in 0..8 {
            let slot = first.wrapping_add(probe) & mask;
            let range = &mut ranges[slot];
            if range.person == row.person || range.event_end == 0 {
                range.person = row.person;
                found = Some(slot);
                break;
            }
        }
        let group = found.unwrap_or_else(|| {
            *overflow.entry(row.person).or_insert_with(|| {
                let index = ranges.len();
                ranges.push(PersonRange {
                    person: row.person,
                    event_end: 0,
                    click_end: 0,
                });
                index
            })
        });
        ranges[group].event_end += 1;
        row.person = group as u64;
    }
    for row in clicks.iter_mut().flatten() {
        let first = first_slot(row.person);
        let mut found = None;
        let mut missed = false;
        for probe in 0..8 {
            let slot = first.wrapping_add(probe) & mask;
            let range = &ranges[slot];
            if range.event_end == 0 {
                missed = true;
                break;
            }
            if range.person == row.person {
                found = Some(slot);
                break;
            }
        }
        if let Some(group) = found.or_else(|| {
            if missed {
                None
            } else {
                overflow.get(&row.person).copied()
            }
        }) {
            ranges[group].click_end += 1;
            row.person = group as u64;
        } else {
            row.person = u64::MAX;
        }
    }
    let mut event_count = 0;
    let mut click_count = 0;
    for range in ranges.iter_mut() {
        let next_event = event_count + range.event_end;
        let next_click = click_count + range.click_end;
        range.event_end = event_count;
        range.click_end = click_count;
        event_count = next_event;
        click_count = next_click;
    }

    if grouped_events.len() < event_count {
        grouped_events.resize(event_count, T::default());
    }
    for row in events.into_iter().flatten() {
        let cursor = &mut ranges[row.person as usize].event_end;
        grouped_events[*cursor] = row.data;
        *cursor += 1;
    }
    if grouped_clicks.len() < click_count {
        grouped_clicks.resize(click_count, Ad::default());
    }
    for row in clicks.into_iter().flatten() {
        if row.person != u64::MAX {
            let cursor = &mut ranges[row.person as usize].click_end;
            grouped_clicks[*cursor] = row.click;
            *cursor += 1;
        }
    }
    rows.clear();
    rows.reserve(event_count / 10);

    let mut event_begin = 0;
    let mut click_begin = 0;
    let bin_count = context
        .end
        .abs_diff(context.start)
        .div_ceil(SESSION_GAP as u64);
    let bin_count = usize::try_from(bin_count).unwrap_or(usize::MAX);
    let use_bins = bin_count.saturating_mul(categories.len().max(1)) <= 131_072;
    for range in ranges.drain(..) {
        if range.event_end == event_begin {
            continue;
        }
        let person = range.person;
        let person_events = &mut grouped_events[event_begin..range.event_end];
        let person_clicks = &mut grouped_clicks[click_begin..range.click_end];
        event_begin = range.event_end;
        click_begin = range.click_end;
        // Identical click copies enter and leave together. Both the latest
        // eligible click and the set of active campaigns are unchanged.
        person_clicks.sort_unstable_by_key(|row| row.time);
        attribution.reset();
        if use_bins && person_events.len() >= bin_count.saturating_mul(2).max(256) {
            bins.record(
                person,
                person_events,
                person_clicks,
                context,
                bin_count,
                attribution,
                totals,
                rows,
                categories,
            );
            continue;
        }
        person_events.sort_unstable();
        let mut current: Option<UserDay> = None;
        let mut at = 0;
        while at < person_events.len() {
            let first = at;
            let start = person_events[first].time(context.start);
            let day = start.div_euclid(DAY) as i32;
            at += 1;
            while at < person_events.len()
                && person_events[at].time(context.start) - person_events[at - 1].time(context.start)
                    <= SESSION_GAP
            {
                at += 1;
            }
            if current.as_ref().is_some_and(|row| row.day != day) {
                finish_day(rows, current.take().unwrap(), categories);
            }
            let row = current.get_or_insert(UserDay {
                person,
                day,
                sessions: 0,
                events: 0,
                revenue: 0,
                category: None,
            });
            let mut revenue = 0i64;
            for event in &person_events[first..at] {
                if let Some((value, category)) = event.purchase() {
                    revenue += i64::from(value);
                    categories[category as usize] += i64::from(value);
                }
            }
            row.sessions += 1;
            row.events += (at - first) as i64;
            row.revenue += revenue;
            let campaign = attribution.advance(person_clicks, start, context);
            totals.record(day, campaign, revenue, &attribution.active);
        }
        if let Some(row) = current {
            finish_day(rows, row, categories);
        }
    }
    output::users(
        &context
            .output
            .join("user_daily")
            .join(format!("part-{shard:03}.parquet")),
        rows,
        &context.people,
        &context.dimensions,
    )?;
    Ok(())
}

fn aggregate<T: GroupEvent>(
    raw: ingest::EventParts<'_>,
    context: &Context,
    started: Instant,
) -> Result<CampaignTotals> {
    // Window end is exclusive. The inclusive last millisecond may be exactly
    // u32::MAX after the lookback origin without losing any timestamp.
    if (context.end - 1).abs_diff(context.click_origin()) <= u64::from(u32::MAX) {
        aggregate_clicks::<T, u32>(raw, context, started)
    } else {
        aggregate_clicks::<T, i64>(raw, context, started)
    }
}

fn aggregate_clicks<T: GroupEvent, C: ClickTime>(
    raw: ingest::EventParts<'_>,
    context: &Context,
    started: Instant,
) -> Result<CampaignTotals> {
    let (events_by_person, clicks_by_person) = rayon::join(
        || ingest::finish_events::<T>(raw, context),
        || ingest::clicks::<C>(context),
    );
    eprintln!(
        "deduplication and clicks {:.3}s",
        started.elapsed().as_secs_f64()
    );
    // Start the largest shards first. A few shoppers can account for a
    // substantial fraction of the input, and their records share a shard.
    let mut jobs: Vec<_> = events_by_person
        .into_iter()
        .zip(clicks_by_person)
        .enumerate()
        .map(|(shard, (events, clicks))| {
            let count = events.iter().map(Vec::len).sum::<usize>();
            let work = count + clicks.iter().map(Vec::len).sum::<usize>();
            (work, shard, count, events, clicks)
        })
        .collect();
    jobs.sort_unstable_by_key(|job| std::cmp::Reverse(job.0));
    let worker_count = jobs.len().min(rayon::current_num_threads());
    let queue = std::sync::Mutex::new(jobs.into_iter());
    let totals = (0..worker_count)
        .into_par_iter()
        .with_max_len(1)
        .map(|_| -> Result<DailyCampaigns> {
            let mut worker = PersonWorker::new(context);
            loop {
                let job = queue.lock().expect("person shard queue poisoned").next();
                let Some((_, shard, count, events, clicks)) = job else {
                    break;
                };
                group_people(events, clicks, shard, context, count, &mut worker)?;
            }
            Ok(worker.totals)
        })
        .try_reduce_with(|left, right| Ok(left.merge(right)))
        .expect("at least one person shard")?
        .into_totals();
    Ok(totals)
}

fn run() -> Result<()> {
    let started = Instant::now();
    let data = PathBuf::from(env::var_os("DATA_DIR").unwrap_or_else(|| "/data".into()));
    let output = PathBuf::from(env::var_os("OUTPUT_DIR").unwrap_or_else(|| "/output".into()));
    let threads: usize = env::var("NCPU")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, usize::from))
        .max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(data.join("manifest.json"))?)?;
    let start = timestamp(
        manifest["window_start"]
            .as_str()
            .ok_or("missing window_start")?
            .as_bytes(),
    )
    .ok_or("bad window_start")?;
    let end = timestamp(
        manifest["window_end"]
            .as_str()
            .ok_or("missing window_end")?
            .as_bytes(),
    )
    .ok_or("bad window_end")?;
    if end <= start {
        return Err("window_end must follow window_start".into());
    }
    let links = Inputs::open(&data.join("links"))?;
    let events = Inputs::open(&data.join("events"))?;
    let clicks = Inputs::open(&data.join("clicks"))?;
    let small = events.bytes() < 1024 * 1024;
    let context = Context {
        click_sources: clicks.chunks(),
        people: People::new(),
        dimensions: Dimensions::load(&data)?,
        start,
        end,
        hash: RandomState::with_seeds(11, 23, 47, 83),
        person_shards: if small {
            1
        } else {
            (threads * 8).next_power_of_two().min(128)
        },
        event_shards: if small { 1 } else { 128 },
        output,
    };
    let (raw, ()) = rayon::join(
        || ingest::scan_events(&events, &context),
        || context.people.load_links(&links),
    );
    eprintln!(
        "events and links {:.3}s; aliases {}",
        started.elapsed().as_secs_f64(),
        context.people.len()
    );
    let totals = if context.end.abs_diff(context.start) <= u64::from(u32::MAX) + 1
        && context.dimensions.categories.len() <= 256
    {
        aggregate::<u64>(raw, &context, started)?
    } else {
        aggregate::<WideEvent>(raw, &context, started)?
    };
    output::campaigns(
        &context.output.join("campaign_daily/part-000.parquet"),
        &totals,
        &context.dimensions,
    )?;
    eprintln!(
        "complete {:.3}s; sessions {}; revenue_cents {}",
        started.elapsed().as_secs_f64(),
        totals.values().map(|row| row.sessions).sum::<i64>(),
        totals.values().map(|row| row.revenue).sum::<i64>()
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("shopping-sessions: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod compact_tests {
    use super::*;

    #[test]
    fn compact_sort_key_preserves_timestamp_price_and_category_extremes() {
        let start = -123_000;
        let row = PersonEvent {
            person: 17,
            ts: start + i64::from(u32::MAX),
            revenue: 10_000_000,
            category: 255,
        };
        let compact = <u64 as GroupEvent>::pack(row, start);
        assert_eq!(compact.time(start), row.ts);
        assert_eq!(compact.purchase(), Some((10_000_000, 255)));
        assert_eq!(
            <u64 as GroupEvent>::from_packed(u32::MAX, (255 << 24) | 10_000_000, start),
            compact
        );
        assert!(<u64 as GroupEvent>::from_packed(0, 0, start)
            .purchase()
            .is_none());
        let wide_row = PersonEvent {
            ts: row.ts + 1,
            category: 256,
            ..row
        };
        let wide = WideEvent::pack(wide_row, start);
        assert_eq!(wide.time(start), wide_row.ts);
        assert_eq!(wide.purchase(), Some((10_000_000, 256)));
    }

    #[test]
    fn compact_click_clock_preserves_both_ends_of_the_window() {
        let origin = -WEEK - 3000;
        assert_eq!(<u32 as ClickTime>::encode(origin, origin), 0);
        assert_eq!(
            <u32 as ClickTime>::encode(origin + i64::from(u32::MAX), origin),
            u32::MAX
        );
        assert_eq!(
            <i64 as ClickTime>::encode(origin + i64::from(u32::MAX) + 1, origin),
            origin + i64::from(u32::MAX) + 1
        );
        assert_eq!(std::mem::size_of::<Ad<u32>>(), 16);
    }
}
