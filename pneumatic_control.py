from redis import Redis
from rq import Queue
from datetime import datetime, timedelta
import serial
import time
import glob
import fire

job_scheduler = Queue(connection=Redis(host='192.168.148.66', port=6379))
TIME_TO_TRIGGER = 2
from control import send_command

def send_signal_job(payload=None):
	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	print(f"> job scheduled at {now}")
	res = job_scheduler.enqueue_in(timedelta(seconds=TIME_TO_TRIGGER), send_command, payload)
	print(res)
	return res


if __name__ == '__main__':
	fire.Fire()
