import json
import asyncio

async def extract_json_data(file_path):
    """
    Asynchronously extract data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary with keys `text`, `tables`, and `figures`.
              `text` contains serialized JSON text as a string.
              `tables` contains the JSON data parsed as a dictionary or list.
    """
    try:
        # Offload blocking file I/O operations to a thread
        async with asyncio.Lock():  # Prevent conflicts when accessing file in async context
            data = await asyncio.to_thread(extract_json_data, file_path)
        return data

    except Exception as e:
        raise ValueError(f"Error extracting data from JSON file asynchronously: {e}")


# Example Usage
if __name__ == "__main__":
    # Synchronous example
    file_path = "data/sample.json"
    try:
        print("Synchronous Extraction:")
        result_sync = extract_json_data(file_path)
        print(result_sync)
    except Exception as e:
        print(e)

    # Asynchronous example
    async def main():
        try:
            print("\nAsynchronous Extraction:")
            result_async = await extract_json_data_async(file_path)
            print(result_async)
        except Exception as e:
            print(e)

    asyncio.run(main())

