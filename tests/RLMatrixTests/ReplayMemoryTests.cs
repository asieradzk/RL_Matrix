using Microsoft.VisualStudio.TestTools.UnitTesting;
using RLMatrix;
using System;
using System.Linq;

namespace RLMatrixTests
{
    [TestClass]
    public class ReplayMemoryTests
    {
        [TestMethod]
        public void TestPushAndSample()
        {
            var replayMemory = new ReplayMemory<float[]>(10, 2);

            replayMemory.Push(new Transition<float[]>(new float[] { 1.0f, 2.0f }, 1, 1.0f, new float[] { 3.0f, 4.0f }));
            replayMemory.Push(new Transition<float[]>(new float[] { 2.0f, 3.0f }, 2, 2.0f, new float[] { 4.0f, 5.0f }));

            Assert.AreEqual(2, replayMemory.Length);

            var samples = replayMemory.Sample();

            Assert.AreEqual(2, samples.Count);
        }

        [TestMethod]
        public void TestMemoryCapacity()
        {
            var replayMemory = new ReplayMemory<float[]>(2, 2);

            replayMemory.Push(new Transition<float[]>(new float[] { 1.0f, 2.0f }, 1, 1.0f, new float[] { 3.0f, 4.0f }));
            replayMemory.Push(new Transition<float[]>(new float[] { 2.0f, 3.0f }, 2, 2.0f, new float[] { 4.0f, 5.0f }));
            replayMemory.Push(new Transition<float[]>(new float[] { 3.0f, 4.0f }, 3, 3.0f, new float[] { 5.0f, 6.0f }));

            // Memory capacity should not exceed 2
            Assert.AreEqual(2, replayMemory.Length);
        }

        [TestMethod]
        public void TestClearMemory()
        {
            var replayMemory = new ReplayMemory<float[]>(10, 2);

            replayMemory.Push(new Transition<float[]>(new float[] { 1.0f, 2.0f }, 1, 1.0f, new float[] { 3.0f, 4.0f }));
            replayMemory.Push(new Transition<float[]>(new float[] { 2.0f, 3.0f }, 2, 2.0f, new float[] { 4.0f, 5.0f }));

            Assert.AreEqual(2, replayMemory.Length);

            replayMemory.ClearMemory();

            Assert.AreEqual(0, replayMemory.Length);
        }

        [TestMethod]
        public void TransposeTest()
        {
            var transitions = new List<Transition<float[]>>()
            {
                new Transition<float[]>(new float[] {1, 2}, 1, 1.5f, new float[] {3, 4}),
                new Transition<float[]>(new float[] {5, 6}, 2, 2.5f, new float[] {7, 8})
            };

            var (states, actions, rewards, nextStates) = transitions.Transpose();

            // Test state
            Assert.AreEqual(1, states[0][0]);
            Assert.AreEqual(2, states[0][1]);
            Assert.AreEqual(5, states[1][0]);
            Assert.AreEqual(6, states[1][1]);

            // Test action
            Assert.AreEqual(1, actions[0]);
            Assert.AreEqual(2, actions[1]);

            // Test reward
            Assert.AreEqual(1.5f, rewards[0]);
            Assert.AreEqual(2.5f, rewards[1]);

            // Test next state
            Assert.AreEqual(3, nextStates[0][0]);
            Assert.AreEqual(4, nextStates[0][1]);
            Assert.AreEqual(7, nextStates[1][0]);
            Assert.AreEqual(8, nextStates[1][1]);
        }
    }
}
