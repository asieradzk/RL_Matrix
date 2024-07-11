using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RLMatrix.Common.Dashboard;

namespace RLMatrix.Dashboard.Services
{
    public interface IExportService
    {
        Task<byte[]> ExportExperimentDataAsCsv(List<ExperimentData> data);
    }

    public class ExportService : IExportService
    {
        public Task<byte[]> ExportExperimentDataAsCsv(List<ExperimentData> data)
        {
            var csv = new StringBuilder();
            csv.AppendLine("Timestamp,Episode,Reward,CumulativeReward,EpisodeLength,ActorLoss,ActorLearningRate,CriticLoss,CriticLearningRate,KLDivergence,Entropy,TargetQValue,Epsilon,TDError,Loss,LearningRate,CategoricalAccuracy,KLDivergenceC51");

            var sortedData = data.OrderBy(d => d.Timestamp).ToList();

            for (int i = 0; i < sortedData.Count; i++)
            {
                var item = sortedData[i];
                csv.AppendLine($"{item.Timestamp},{i + 1},{item.Reward},{item.CumulativeReward},{item.EpisodeLength},{item.ActorLoss},{item.ActorLearningRate},{item.CriticLoss},{item.CriticLearningRate},{item.KLDivergence},{item.Entropy},{item.TargetQValue},{item.Epsilon},{item.TDError},{item.Loss},{item.LearningRate},{item.CategoricalAccuracy},{item.KLDivergenceC51}");
            }

            return Task.FromResult(Encoding.UTF8.GetBytes(csv.ToString()));
        }
    }
}