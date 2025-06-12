import asyncio, aiohttp, random, time

API_URL = "http://localhost:8000/predict"
SAMPLE_TEXTS = [
    "I love this product! It's absolutely amazing.",
    "Terrible! I want my money back.",
    "This is the best service I've used so far.",
    "Not what I expected, pretty bad.",
    "Fantastic experience, would recommend to everyone.",
]

ERROR_PAYLOADS = [
    {},  # thiếu key text
    {"text": ""},  # văn bản trống
]

CONCURRENCY = 20
REQUESTS_PER_WORKER = 100

async def send(session):
    for _ in range(REQUESTS_PER_WORKER):
        if random.random() < 0.1:  # 10% request lỗi
            payload = random.choice(ERROR_PAYLOADS)
        else:
            payload = {"text": random.choice(SAMPLE_TEXTS)}
        try:
            async with session.post(API_URL, json=payload, timeout=5) as resp:
                await resp.text()
        except Exception:
            pass  # lỗi sẽ được ghi log tại server

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(send(session)) for _ in range(CONCURRENCY)]
        start = time.time()
        await asyncio.gather(*tasks)
        print(f"Sent {CONCURRENCY*REQUESTS_PER_WORKER} requests in {time.time()-start:.2f}s")

if __name__ == "_main_":
    asyncio.run(main()) 