// Sends Meta Quest 2 controller poses to the Python bridge.
//
// This is a custom scene for the Blender bench. It is not the official
// ReachyTeleoperation project, which must stay on Unity 2020.3 because of its
// gRPC plugin. Copy this folder into a Unity 2022.3 project, enable the Oculus
// (or OpenXR) plug-in for Android, and put the script on an empty GameObject.
//
// Buttons, matching the Reachy 2023 Quest map where they can:
//   A  right primary    calibrate the body yaw. Stand still. Do not move your feet after this.
//   B  right secondary  emergency stop
//   Y  left secondary   start or stop a recording episode
//   right index trigger close the right gripper (0 open, 1 closed)
//
// Positions are Unity tracking space: X right, Y up, Z forward, meters.

using System.Net.Sockets;
using System.Text;
using UnityEngine;
using UnityEngine.XR;

namespace ReachyPour
{
    public class QuestTeleopPublisher : MonoBehaviour
    {
        [Tooltip("IP of the computer running `reachy-pour serve` or `reachy-pour record --mode quest`. Not the robot IP.")]
        public string host = "192.168.0.20";
        public int port = 29023;
        [Range(10f, 90f)] public float sendHz = 30f;

        UdpClient udp;
        float sendAccumulator;
        readonly StringBuilder builder = new StringBuilder(512);

        void OnEnable()
        {
            udp = new UdpClient();
        }

        void OnDisable()
        {
            if (udp != null)
            {
                udp.Close();
                udp = null;
            }
        }

        void Update()
        {
            sendAccumulator += Time.unscaledDeltaTime;
            if (sendAccumulator < 1f / sendHz)
                return;
            sendAccumulator = 0f;
            Send();
        }

        void Send()
        {
            if (udp == null)
                return;

            ReadNode(XRNode.Head, out Vector3 headPos, out Quaternion headRot, out _, out _);
            ReadNode(XRNode.RightHand, out Vector3 rightPos, out Quaternion rightRot, out float rightTrig, out _);
            ReadNode(XRNode.LeftHand, out Vector3 leftPos, out Quaternion leftRot, out float leftTrig, out _);
            bool a = Button(XRNode.RightHand, CommonUsages.primaryButton);
            bool b = Button(XRNode.RightHand, CommonUsages.secondaryButton);
            bool x = Button(XRNode.LeftHand, CommonUsages.primaryButton);
            bool y = Button(XRNode.LeftHand, CommonUsages.secondaryButton);

            builder.Clear();
            builder.Append("{\"type\":\"quest2\"");
            AppendInt(builder, "a", a);
            AppendInt(builder, "b", b);
            AppendInt(builder, "x", x);
            AppendInt(builder, "y", y);
            AppendPose(builder, "h", headPos, headRot);
            AppendPose(builder, "r", rightPos, rightRot);
            builder.Append(",\"rtrig\":").Append(rightTrig.ToString("F3"));
            AppendPose(builder, "l", leftPos, leftRot);
            builder.Append(",\"ltrig\":").Append(leftTrig.ToString("F3"));
            builder.Append('}');

            byte[] payload = Encoding.UTF8.GetBytes(builder.ToString());
            udp.Send(payload, payload.Length, host, port);
        }

        static void ReadNode(XRNode node, out Vector3 position, out Quaternion rotation, out float trigger, out float grip)
        {
            position = Vector3.zero;
            rotation = Quaternion.identity;
            trigger = 0f;
            grip = 0f;
            var device = InputDevices.GetDeviceAtXRNode(node);
            if (!device.isValid)
                return;
            device.TryGetFeatureValue(CommonUsages.devicePosition, out position);
            device.TryGetFeatureValue(CommonUsages.deviceRotation, out rotation);
            device.TryGetFeatureValue(CommonUsages.trigger, out trigger);
            device.TryGetFeatureValue(CommonUsages.grip, out grip);
        }

        static bool Button(XRNode node, InputFeatureUsage<bool> usage)
        {
            var device = InputDevices.GetDeviceAtXRNode(node);
            return device.isValid && device.TryGetFeatureValue(usage, out bool value) && value;
        }

        static void AppendInt(StringBuilder builder, string name, bool value)
        {
            builder.Append(",\"").Append(name).Append("\":").Append(value ? '1' : '0');
        }

        static void AppendPose(StringBuilder builder, string prefix, Vector3 position, Quaternion rotation)
        {
            builder.Append(",\"").Append(prefix).Append("x\":").Append(position.x.ToString("F4"));
            builder.Append(",\"").Append(prefix).Append("y\":").Append(position.y.ToString("F4"));
            builder.Append(",\"").Append(prefix).Append("z\":").Append(position.z.ToString("F4"));
            builder.Append(",\"").Append(prefix).Append("qx\":").Append(rotation.x.ToString("F4"));
            builder.Append(",\"").Append(prefix).Append("qy\":").Append(rotation.y.ToString("F4"));
            builder.Append(",\"").Append(prefix).Append("qz\":").Append(rotation.z.ToString("F4"));
            builder.Append(",\"").Append(prefix).Append("qw\":").Append(rotation.w.ToString("F4"));
        }
    }
}
