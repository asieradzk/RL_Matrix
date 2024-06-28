using System;
using System.Collections.Generic;

namespace RLMatrix
{
    public class SumTree
    {
        public float[] tree;
        private int capacity;
        private int count;
        public float MaxPriority { get; private set; } = 1f;

        public SumTree(int capacity)
        {
            this.capacity = capacity;
            this.tree = new float[2 * capacity];
            this.count = 0;
        }

        public void Update(int index, float priority)
        {
            int treeIndex = index + capacity; // Translate index to tree index
            float change = priority - tree[treeIndex];
            tree[treeIndex] = priority;

            if (priority > MaxPriority)
            {
                MaxPriority = priority;
            }

            // Update parent nodes
            while (treeIndex > 0)
            {
                treeIndex /= 2; // Move up the tree
                tree[treeIndex] += change; // Update the sum
            }
        }
        public int GetMinPriorityIndex()
        {
            int minPriorityIndex = 0;
            float minPriority = float.MaxValue;

            for (int i = capacity; i < 2 * capacity; i++)
            {
                if (tree[i] < minPriority)
                {
                    minPriority = tree[i];
                    minPriorityIndex = i - capacity;
                }
            }

            return minPriorityIndex;
        }
        public int GetMaxPriorityIndex()
        {
            int maxPriorityIndex = 0;
            float maxPriority = float.MinValue;

            for (int i = capacity; i < 2 * capacity; i++)
            {
                if (tree[i] > maxPriority)
                {
                    maxPriority = tree[i];
                    maxPriorityIndex = i - capacity;
                }
            }

            return maxPriorityIndex;
        }

        public int Retrieve(float value)
        {
            int index = 1; // Start from the root
            while (index < capacity) // Ensure index is within leaf node range
            {
                index *= 2; // Move to left child by default
                if (value >= tree[index])
                {
                    value -= tree[index]; // Deduct the value of the left child when moving right
                    index++; // Move to right child
                }
            }
            return Math.Min(index - capacity, count - 1); // Ensure index is within bounds
        }

        public float TotalSum => tree[1]; // Total priority sum is at the root

        public int Add(float priority)
        {
            // Instead of throwing an exception, recycle the oldest entry
            int index = count % capacity; // Use modulo to create a circular buffer effect
            Update(index, priority); // Update or add the priority in the SumTree
            if (count < capacity) count++; // Increment count until the tree is full
            return index;
        }

        public void Clear()
        {
#if NET8_0_OR_GREATER
            Array.Fill(tree, 0f); // Reset all tree elements to 0
#else
            for (int i = 0; i < tree.Length; i++)
            {
                tree[i] = 0f;
            }
#endif
            count = 0; // Reset countl
            MaxPriority = 1.0f; // Reset MaxPriority
        }
    
    }
}
