

# --- The event queue ---#
import heapq
import numpy as np
import time
import threading


# constantly generates stimulus from the environment


# simulate the brain
class BrainSimulatorThread(threading.Thread):
    def __init__(self, event_queue: PriorityQueue):
        super().__init__()
        self.event_queue = event_queue
        self.model = Abyssol(10000, 1000) # Your real vectorized model
        self.current_sim_time = 0.0
        self.running = True

    def process_event(self, event):
        # Your logic for handling an event and maybe creating new internal events
        event_time, event_type, target_index = event
        # print(f"  [SIMULATOR]: Processing {event_type} at time {event_time:.2f}")
        pass

    def run(self):
        """The main simulation loop."""
        print("[SIMULATOR THREAD]: Started and waiting for events.")
        while self.running:
            try:
                # Get an event from the shared queue.
                # The 'timeout=0.01' is crucial. It means:
                # "Wait for an event, but if nothing arrives in 10ms,
                # wake up anyway." This prevents the loop from getting
                # stuck if there are no events.
                event = self.event_queue.get(timeout=0.01)
                
                # We got an event! Process it.
                self.process_event(event)
                
            except queue.Empty:
                # This is not an error! It just means no events were ready.
                # We can do housekeeping work here, or just continue.
                # print("[SIMULATOR]: Queue is empty, sleeping...")
                pass
    
    def stop(self):
        print("[SIMULATOR THREAD]: Stopping...")
        self.running = False
        

if __name__ == "__main__":
    # 1. Create the shared, thread-safe queue
    shared_queue = queue.PriorityQueue()
    
    # 2. Define the stimulus mapping
    key_map = {'a': [1, 2, 3], 's': [4, 5, 6]}

    # 3. Create the thread instances
    producer = StimulusProducerThread(shared_queue, key_map)
    simulator = BrainSimulatorThread(shared_queue)

    print("--- Starting threads (Press 'a' or 's' to generate stimuli) ---")
    
    # 4. Start the threads. They will now run in parallel.
    producer.start()
    simulator.start()

    try:
        # The main thread can do other work, or just wait.
        # Here, we'll just keep it alive until the user presses Ctrl+C.
        while True:
            time.sleep(1)
            print(f"Main Thread: Queue size is currently {shared_queue.qsize()}")
            
    except KeyboardInterrupt:
        print("\n--- Main thread received shutdown signal ---")
        
    finally:
        # 5. Tell the threads to stop and wait for them to finish cleanly.
        producer.stop()
        simulator.stop()
        
        producer.join() # Wait for producer thread to exit
        simulator.join()# Wait for simulator thread to exit
        
        print("--- All threads have been stopped. Exiting. ---")