from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from src.config import Config
from src.monitoring.worker import Worker


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.last_processed_minute = None
        self.last_alerted_bucket = {}

    def run(self):
        while True:
            current_time = datetime.now()

            if self._should_process(current_time):
                self._process_cycle()
                self.last_processed_minute = current_time.replace(second=0, microsecond=0)

            sleep(0.1)

    def _should_process(self, current_time: datetime) -> bool:
        if current_time.minute % 15 != 0:
            return False

        if current_time.second < self.config.offset_seconds:
            return False

        current_minute = current_time.replace(second=0, microsecond=0)
        if self.last_processed_minute == current_minute:
            return False

        return True

    def _process_cycle(self):
        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            futures = [
                executor.submit(self._process_token, token)
                for token in self.config.tokens
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error: {e}")

    def _process_token(self, token: str):
        worker = Worker(self.config, token, self.last_alerted_bucket)
        worker.process()
