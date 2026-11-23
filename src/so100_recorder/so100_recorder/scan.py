"""Hardware check: read every servo, then wiggle each joint once around its resting position."""
import time

import serial

from so100_recorder import config
from so100_recorder.feetech import enable_torque, read_position, write_position

JOINT_NAMES = {
    1: 'Shoulder Pan',
    2: 'Shoulder Lift',
    3: 'Elbow Flex',
    4: 'Wrist Flex',
    5: 'Wrist Roll',
    6: 'Gripper',
}

WIGGLE_TICKS = 150  # About 13 degrees
WIGGLE_PAUSE = 0.4


def main():
    try:
        ser = serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=0.05)
        print(f'Connected to {config.SERIAL_PORT}')
    except Exception as e:
        print(f'Failed to connect: {e}')
        return

    # 1. Read resting positions, so the arm doesn't snap when torque comes on.
    print('\nReading current resting positions...')
    home_positions = {}
    for servo_id, name in JOINT_NAMES.items():
        pos = read_position(ser, servo_id, settle=0.005)
        if pos is None:
            print(f'Error: could not read servo {servo_id}. Aborting for safety!')
            return
        home_positions[servo_id] = pos
        print(f'   {name} rests at: {pos}')

    # 2. Hold the current position.
    print('\nLocking arm at current position (torque ON)...')
    for servo_id in JOINT_NAMES:
        write_position(ser, servo_id, home_positions[servo_id])
        enable_torque(ser, servo_id, True)
        time.sleep(0.05)

    print('STAND BACK! Starting wiggle sequence in 3 seconds...')
    time.sleep(3)

    # 3. Wiggle each joint: forward, backward past center, back to center.
    for servo_id, name in JOINT_NAMES.items():
        print(f'Moving {name}...')
        base_pos = home_positions[servo_id]
        for target in (base_pos + WIGGLE_TICKS, base_pos - WIGGLE_TICKS, base_pos):
            write_position(ser, servo_id, target)
            time.sleep(WIGGLE_PAUSE)

    # 4. Relax.
    print('\nSequence complete. Relaxing motors (torque OFF)...')
    for servo_id in JOINT_NAMES:
        enable_torque(ser, servo_id, False)

    print('You can now freely move the arm again!')
    ser.close()


if __name__ == '__main__':
    main()
