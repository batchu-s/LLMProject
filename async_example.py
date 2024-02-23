import asyncio

async def count():
    print("One")
    await asyncio.sleep(1) # waits for 1 second: an example of blocking I/O
    print("Two")

async def main():
    await asyncio.gather(count(), count(), count())

if __name__ == "__main__":
    import time
    s = time.perf_counter()
    asyncio.run(main()) # our main entry point
    elapsed = time.perf_counter() - s
    print(f"Script executed in {elapsed:0.2f} seconds.")