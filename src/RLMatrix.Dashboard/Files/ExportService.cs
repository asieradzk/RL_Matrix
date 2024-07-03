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
            csv.AppendLine("Timestamp,Episode,Loss,Reward,LearningRate,CumulativeReward,EpisodeLength");

            // Sort the data by timestamp to ensure correct episode numbering
            var sortedData = data.OrderBy(d => d.Timestamp).ToList();

            for (int i = 0; i < sortedData.Count; i++)
            {
                var item = sortedData[i];
                csv.AppendLine($"{item.Timestamp},{i + 1},{item.Loss},{item.Reward},{item.LearningRate},{item.CumulativeReward},{item.EpisodeLength}");
            }

            return Task.FromResult(Encoding.UTF8.GetBytes(csv.ToString()));
        }
    }
}