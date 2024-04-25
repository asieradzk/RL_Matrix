using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLMatrix
{

    public class BaseRolloutAgent<TState>
    {
        protected readonly Dictionary<Guid, IEnvironmentAsync<TState>> _environments;
        protected readonly Dictionary<Guid, Episode<TState>> _ennvGuidPairs;
        protected readonly IDiscreteProxy<TState> _agent;
        public DQNAgentOptions _options;

        public BaseRolloutAgent(DQNAgentOptions options, IEnumerable<IEnvironmentAsync<TState>> environments)
        {
            _environments = environments.ToDictionary(env => Guid.NewGuid(), env => env);
            _ennvGuidPairs = new();
            foreach (var env in _environments)
            {
                _ennvGuidPairs[env.Key] = new Episode<TState>();
            }
            _agent = CreateAgent(options, environments.FirstOrDefault());
            _options = options;
        }

        protected IDiscreteProxy<TState> CreateAgent(DQNAgentOptions options, IEnvironmentAsync<TState> env)
        {
            return new LocalDiscreteAgent<TState>(options, env.actionSize, env.stateSize);
        }
        List<double> chart = new();
        public async Task Step(bool isTraining = true)
        {
            //GETS INITIAL STATES FOR ALL ENVS
            List<Task<(Guid environmentId, TState state)>> stateTaskList = new();
            foreach (var env in _environments)
            {
                var stateTask = GetStateAsync(env.Key, env.Value);
                stateTaskList.Add(stateTask);
            }
            var stateResults = await Task.WhenAll(stateTaskList);


            //GETS ACTIONS FOR ALL DETERMINED STATES
            List<(Guid environmentId, TState state)> payload = stateResults.ToList();
            var actions = await _agent.SelectActionsBatchAsync(payload);



            //STEPS ALL ENVS AND GETS REWARDS AND DONES
            List<Task<(Guid environmentId, (float, bool) reward)>> rewardTaskList = new();
            foreach (var action in actions)
            {
                var env = _environments[action.Key];
                var rewardTask = env.Step(action.Value)
                    .ContinueWith(t => (action.Key, t.Result));  // Ensure that the task returns a tuple with the Guid
                rewardTaskList.Add(rewardTask);
            }

            await Task.WhenAll(rewardTaskList);


            //GETS NEXT STATES FOR ALL ENVS
            List<Task<(Guid environmentId, TState state)>> nextStateTaskList = new();
            foreach (var env in _environments)
            {
                var stateTask = GetStateAsync(env.Key, env.Value);
                nextStateTaskList.Add(stateTask);
            }
            var nextStateResults = await Task.WhenAll(nextStateTaskList);

            //Process episodes
            ConcurrentBag<TransitionPortable<TState>> transitionsToShip = new();
            ConcurrentBag<double> rewards = new();
            foreach (var env in _environments)
            {
                var key = env.Key;
                var episode = _ennvGuidPairs[key];
                var state = stateResults.First(x => x.environmentId == key).state;
                var action = actions[key];
                var stepResult = rewardTaskList.First(x => x.Result.environmentId == key).Result;
                var reward = stepResult.reward.Item1;
                var isDone = stepResult.reward.Item2;
                var nextState = nextStateResults.First(x => x.environmentId == key).state;
                episode.AddTransition(state, isDone, action, reward);
                if (isDone)
                {
                    foreach (var transition in episode.CompletedEpisodes)
                    {
                        transitionsToShip.Add(transition);
                    }
                    rewards.Add(episode.cumulativeReward);
                    episode.CompletedEpisodes.Clear();
                }
            }

            if (_options.DisplayPlot != null)
            {
                chart.AddRange(rewards);
                _options.DisplayPlot.CreateOrUpdateChart(chart);
            }

            await _agent.UploadTransitionsAsync(transitionsToShip.ToList());

            
            await _agent.OptimizeModelAsync();

        }


        public async ValueTask<Dictionary<Guid, int[]>> GetActionsBatchAsync(List<(Guid environmentId, TState state)> stateInfos)
        {
            return await _agent.SelectActionsBatchAsync(stateInfos);
        }
        private async Task<(Guid environmentId, TState state)> GetStateAsync(Guid environmentId, IEnvironmentAsync<TState> env)
        {
            var state = await env.GetCurrentState();
            return (environmentId, state);
        }


        public class Episode<T>
        {
            Guid? guidCache;
            public float cumulativeReward = 0;
            private List<TransitionPortable<T>> TempBuffer = new();
            public List<TransitionPortable<T>> CompletedEpisodes = new();

            public void AddTransition(T state, bool isDone, int[] discreteACtions, float reward)
            {
                Guid? nextGuid = null;
                if(guidCache == null)
                {
                    guidCache = Guid.NewGuid();
                    cumulativeReward = 0;
                }

                if(!isDone)
                {
                    nextGuid = Guid.NewGuid();
                }
                var transition = new TransitionPortable<T>((Guid)guidCache, DeepCopy(state), discreteACtions, null, reward, nextGuid);
                TempBuffer.Add(transition);
                cumulativeReward += reward;
                guidCache = nextGuid;

                if(isDone)
                {
                    CompletedEpisodes.AddRange(TempBuffer);
                    TempBuffer.Clear();
                }


                T DeepCopy(T input)
                {
                    if (!typeof(T).IsArray)
                    {
                        throw new InvalidOperationException("This method can only be used with arrays!");
                    }

                    // Handle nulls
                    if (ReferenceEquals(input, null))
                    {
                        return default(T);
                    }

                    var rank = ((Array)(object)input).Rank;
                    var lengths = new int[rank];
                    for (int i = 0; i < rank; ++i)
                        lengths[i] = ((Array)(object)input).GetLength(i);

                    var clone = Array.CreateInstance(typeof(T).GetElementType(), lengths);

                    Array.Copy((Array)(object)input, clone, ((Array)(object)input).Length);

                    return (T)(object)clone;
                }
            }
        }

    }
}
