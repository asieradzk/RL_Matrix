namespace RLMatrix.Agents.Common
{

    public class Episode<T>
    {
        Guid? guidCache;
        public float cumulativeReward = 0;
        private List<TransitionPortable<T>> TempBuffer = new();
        public List<TransitionPortable<T>> CompletedEpisodes = new();

        public void AddTransition(T state, bool isDone, int[] discreteActions, float reward)
        {
            Guid? nextGuid = null;
            if (guidCache == null)
            {
                guidCache = Guid.NewGuid();
                cumulativeReward = 0;
            }

            if (!isDone)
            {
                nextGuid = Guid.NewGuid();
            }
            var transition = new TransitionPortable<T>((Guid)guidCache, state, discreteActions, new float[0], reward, nextGuid);
            TempBuffer.Add(transition);
            cumulativeReward += reward;
            guidCache = nextGuid;

            if (isDone)
            {
                CompletedEpisodes.AddRange(TempBuffer);
                TempBuffer.Clear();
            }
        }
    }
}
