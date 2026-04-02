---
title: "Asyncio in Python"
date: 2026-03-24
tags: ["python", "asyncio"]
author: "Ryan H."
description: "This blog post covers the asyncio in python."
summary: "This blog post covers the asyncio in python."
cover:
    image: "asyncio-in-python.png"
    alt: "Asyncio in Python"
    relative: true
---

## Introduction


await-able objects:
- corroutines: when calling async function, it returns a coroutine object, calling a coroutine return an await-able object, but it does not run yet. Only when `await` will run the coroutine and return the result.
- tasks: when calling `asyncio.create_task`, it returns a task object, wrapped around a coroutine that schedules the coroutine to run on the event loop. useful to run multiple coroutines in parallel. Also task add functions to interact with the task, e.g. `cancel()` to cancel the task.
- futures: low-level await-able objects, used to represent an eventual result


# Asyncio basics

## sleep
```python
import asyncio

async def sleep_for_seconds(seconds):
    await asyncio.sleep(seconds)
```

## async generator

```python
import asyncio

async def async_generator():
    for i in range(10):
        # await something async
        yield i

async def async_generator_example():
    async for i in async_generator():
        print(i)
```

## async context manager
```python
import asyncio

@asynccontextmanager
async def async_context_manager():
    try:
        print("entering context manager")
        # do something async
        yield
    finally:
        # cleanup
        print("exiting context manager")

async def async_context_manager_example():
    async with async_context_manager() as cm:
        print(cm)
```
Alternatively, you can implement it as a class with __aenter__ and __aexit__:
```python
class AsyncContextManager:
    async def __aenter__(self):
        print("entering")
        return "resource"

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("exiting")

async def async_context_manager_example():
    async with AsyncContextManager() as cm:
        print(cm)
```

## wait for multiple tasks with TaskGroup

asyncio.gather is a function that waits for multiple tasks to complete and returns a list of results.
```python
import asyncio

async def coro(i):
    ...

async def gather_example():
    # gather coroutines
    coros = [coro(i)) for i in range(10)]
    
    # return_exceptions=True, all coros run to completion, even if one raises an exception
    # results are either the result or the exception
    # by default, return_exceptions=False, if one coro raises an exception, the other coros become orphaned
    results = await asyncio.gather(*coros, return_exceptions=True)
    print(f"results: {results}")

    # gather tasks
    tasks = [asyncio.create_task(coro(i)) for i in range(10)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"results: {results}")

```

TaskGroup is a context manager that creates a group of tasks and waits for them to complete.
It handles the cancellation of tasks when the context manager is exited. If one task raises an exception, all other tasks will be cancelled.
```python
import asyncio

async def coro(i):
    ...

async def wait_for_multiple_tasks(coros):
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(coro(i)) for i in range(10)]
        # All tasks are awaited when the context manager exits

    return [t.result() for t in tasks]
```

## use sync code in async functions
delegate to threads
```python
async def sync_to_async(sync_function, *args, **kwargs):
    task = asyncio.create_task(asyncio.to_thread(sync_function, *args, **kwargs))
    return await task
```

delegate to sub-processes
```python
from concurrent.futures import ProcessPoolExecutor
import asyncio

async def delegate_to_subprocess(sync_function, *args, **kwargs):
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as executor:
        task = loop.run_in_executor(executor, sync_function, *args, **kwargs)
        return await task
```


## use async code in sync functions

```python
def async_to_sync(async_function):
    return asyncio.run(async_function())
```


# Asyncio libraries

## networking requests
```python
import asyncio
import aiohttp

async def fetch_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

## file operations
```python
import asyncio
import aiofiles

async def read_file(file_path):
    async with aiofiles.open(file_path, mode='r') as file:
        return await file.read()
```