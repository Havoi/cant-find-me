"""Generate synthetic CSVs for compiler profile collection during the image build."""
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import random
import sys


def generate(root, event_count=650_000):
    rng = random.Random(8873)
    start = datetime(2027, 3, 1, tzinfo=timezone.utc)
    epoch_ms = int(start.timestamp()) * 1000
    day = 86_400
    window = 14 * day
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"

    def suffix():
        number = rng.getrandbits(40)
        out = []
        while number:
            number, digit = divmod(number, 36)
            out.append(digits[digit])
        return "".join(reversed(out)) or "0"

    def timestamp(seconds, form):
        if form >= 7:
            return str(epoch_ms + seconds * 1000 + rng.randrange(1000))
        local = start + timedelta(seconds=seconds)
        if form < 6:
            return local.strftime("%Y-%m-%dT%H:%M:%SZ")
        minutes = rng.choice((-480, -330, 90, 345, 600))
        local += timedelta(minutes=minutes)
        return (local.strftime("%Y-%m-%dT%H:%M:%S")
                + f"{'+' if minutes > 0 else '-'}{abs(minutes)//60:02d}:{abs(minutes)%60:02d}")

    def write(files, header, rows):
        with ExitStack() as stack:
            streams = []
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = stack.enter_context(path.open("w", encoding="utf-8", newline="\n"))
                stream.write(header + "\n")
                streams.append(stream)
            weights = [i for i in range(len(streams)) for _ in range(max(1, 30 // (i + 1)))]
            for row in rows:
                stream = streams[0] if len(streams) == 1 else streams[rng.choice(weights)]
                stream.write(",".join(f'"{field}"' if rng.randrange(5) == 0 else str(field)
                                      for field in row) + "\n")

    def parts(name, count):
        return [root / name / f"part-{i:02d}.csv" for i in range(count)]

    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({
        "window_start": timestamp(0, 0),
        "window_end": timestamp(window, 0)
    }), encoding="utf-8")
    people = max(40, event_count // 30)
    aliases, anchors, links = [], [], []
    for person in range(people):
        q = rng.randrange(100)
        count = 1 if q < 45 else 2 if q < 75 else 3 if q < 88 else rng.randint(4, 8)
        if person < 5:
            count = rng.randint(3, 6)
        if person == 5:
            count = 140
        ids = ["u" + suffix() for _ in range(count)]
        aliases.append(ids)
        for i in range(1, count):
            links.append((ids[i], ids[i - 1] if person == 5 else ids[rng.randrange(i)]))
        # Sample around session starts rather than choosing each row's time independently.
        anchor_count = 200 if person < 5 else rng.randint(1, 10)
        anchors.append([rng.randrange(window - 900) for _ in range(anchor_count)])
    rng.shuffle(links)

    def shopper():
        if rng.randrange(14) == 0:
            return rng.randrange(5)
        return 5 + int((rng.random() ** 2) * (people - 5))

    def identity(person):
        ids = aliases[person]
        return ids[0] if rng.randrange(2) == 0 else rng.choice(ids)

    def event_rows():
        for i in range(event_count):
            person = shopper()
            second = rng.choice(anchors[person]) + rng.randrange(900)
            kind = rng.randrange(100)
            event_type = "purchase" if kind < 3 else "cart" if kind < 13 else "view"
            price = rng.randint(100, 50_000) if kind < 3 or rng.randrange(2) else ""
            arrival = second + rng.randrange(1200) + (day if i % 97 == 0 else 0)
            ts = timestamp(second, rng.randrange(10)) if i % 211 else "bad"
            row = ["e"+suffix(), identity(person), ts, timestamp(arrival, 0),
                   event_type, f"p{rng.randrange(1, 50_001)}", price]
            yield row
            if i % 31 == 0:
                row[3] = timestamp(arrival + rng.randint(1, 600), 0)
                yield row

    def click_rows():
        for i in range(event_count // 4):
            person = shopper()
            second = rng.choice(anchors[person]) - rng.randrange(7 * day)
            campaign = rng.randrange(4) if rng.randrange(10) < 3 else rng.randrange(2000)
            row = ("k"+suffix(), identity(person), f"c_{campaign}", timestamp(second, rng.randrange(10)))
            yield row
            if i % 53 == 0:
                yield row

    write(parts("events", 24), "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents", event_rows())
    write(parts("clicks", 8), "click_id,user_id,campaign_id,ts", click_rows())
    write(parts("links", 4), "user_id_a,user_id_b", links)
    write([root / "products.csv"], "product_id,category",
          ((f"p{i}", f"category_{i%20}") for i in range(1, 50_001)))
    write([root / "campaigns.csv"], "campaign_id,channel",
          ((f"c_{i}", f"channel_{i%5}") for i in range(2000)))


if __name__ == "__main__":
    generate(Path(sys.argv[1]))
