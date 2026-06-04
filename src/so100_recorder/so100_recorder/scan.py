import serial
import time

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 1000000

def calc_checksum(packet):
    return (~sum(packet[2:]) & 0xFF)

def read_position(ser, servo_id):
    """Reads the current position so we don't violently snap the arm"""
    packet = [0xFF, 0xFF, servo_id, 4, 0x02, 0x38, 2]
    packet.append(calc_checksum(packet))
    ser.reset_input_buffer()
    ser.write(bytearray(packet))
    time.sleep(0.005)
    
    resp = ser.read(8)
    if len(resp) == 8 and resp[0] == 0xFF and resp[1] == 0xFF:
        return (resp[6] << 8) | resp[5]
    return None

def enable_torque(ser, servo_id, enable=True):
    """Turns the motor stiffness ON (1) or OFF (0)"""
    val = 1 if enable else 0
    packet = [0xFF, 0xFF, servo_id, 4, 0x03, 0x28, val] # 0x28 is Torque Enable Register
    packet.append(calc_checksum(packet))
    ser.write(bytearray(packet))

def set_position(ser, servo_id, position):
    """Commands the servo to move to a specific 0-4096 value"""
    # Clamp safety limits
    position = max(0, min(4095, position))
    
    pos_l = position & 0xFF
    pos_h = (position >> 8) & 0xFF
    packet = [0xFF, 0xFF, servo_id, 5, 0x03, 0x2A, pos_l, pos_h] # 0x2A is Goal Position Register
    packet.append(calc_checksum(packet))
    ser.write(bytearray(packet))

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
        print(f"🔌 Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"🚨 Failed to connect: {e}")
        return

    joint_names = {
        1: "Shoulder Pan", 2: "Shoulder Lift", 3: "Elbow Flex", 
        4: "Wrist Flex", 5: "Wrist Roll", 6: "Gripper"
    }

    # --- 1. Read Current Resting Positions ---
    print("\n🔍 Reading current resting positions...")
    home_positions = {}
    for i in range(1, 7):
        pos = read_position(ser, i)
        if pos is None:
            print(f"🚨 Error: Could not read Servo {i}. Aborting for safety!")
            return
        home_positions[i] = pos
        print(f"   {joint_names[i]} rests at: {pos}")

    # --- 2. Lock the arm in place ---
    print("\n🔒 Locking arm at current position (Torque ON)...")
    for i in range(1, 7):
        set_position(ser, i, home_positions[i]) # Tell it to stay exactly where it is
        enable_torque(ser, i, True)             # Turn on the motor
        time.sleep(0.05)

    print("⚠️ STAND BACK! Starting wiggle sequence in 3 seconds...")
    time.sleep(3)

    # --- 3. The Wiggle Loop ---
    wiggle_amount = 150 # About 13 degrees of movement

    for i in range(1, 7):
        print(f"🚀 Moving {joint_names[i]}...")
        base_pos = home_positions[i]
        
        # Move forward
        set_position(ser, i, base_pos + wiggle_amount)
        time.sleep(0.4)
        
        # Move backward past center
        set_position(ser, i, base_pos - wiggle_amount)
        time.sleep(0.4)
        
        # Return to exact center
        set_position(ser, i, base_pos)
        time.sleep(0.4)

    # --- 4. Relax the arm ---
    print("\n✅ Sequence complete. Relaxing motors (Torque OFF)...")
    for i in range(1, 7):
        enable_torque(ser, i, False)
        
    print("🎉 You can now freely move the arm again!")
    ser.close()

if __name__ == '__main__':
    main()