"""Minimal FeeTech STS3215 serial protocol helpers."""
import time

HEADER = [0xFF, 0xFF]

INSTR_READ = 0x02
INSTR_WRITE = 0x03

REG_TORQUE_ENABLE = 0x28
REG_GOAL_POSITION = 0x2A
REG_PRESENT_POSITION = 0x38

MAX_TICKS = 4095


def checksum(packet):
    """Inverted low byte of the sum of everything after the header."""
    return ~(sum(packet[2:]) & 0xFF) & 0xFF


def _send(ser, packet):
    packet.append(checksum(packet))
    ser.write(bytearray(packet))


def enable_torque(ser, motor_id, enable=True):
    """Turn motor stiffness on or off."""
    _send(ser, HEADER + [motor_id, 0x04, INSTR_WRITE, REG_TORQUE_ENABLE, 1 if enable else 0])


def write_position(ser, motor_id, position_ticks):
    """Command a goal position, clamped to the servo's 0-4095 range."""
    ticks = max(0, min(int(position_ticks), MAX_TICKS))
    _send(ser, HEADER + [motor_id, 0x05, INSTR_WRITE, REG_GOAL_POSITION,
                         ticks & 0xFF, (ticks >> 8) & 0xFF])


def read_position(ser, motor_id, settle=0.0):
    """Read the present position in ticks, or None if the reply is invalid."""
    ser.reset_input_buffer()
    _send(ser, HEADER + [motor_id, 0x04, INSTR_READ, REG_PRESENT_POSITION, 0x02])
    if settle:
        time.sleep(settle)
    resp = ser.read(8)
    if len(resp) == 8 and resp[0] == 0xFF and resp[1] == 0xFF:
        return resp[5] | (resp[6] << 8)
    return None
