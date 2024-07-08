using Microsoft.AspNetCore.SignalR;
using RLMatrix.Common.Dashboard;
using RLMatrix.Dashboard;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace RLMatrix.Dashboard.Hubs
{
    public class ExperimentDataHub : Hub
    {
        private readonly IDashboardService _dashboardService;

        public ExperimentDataHub(IDashboardService dashboardService)
        {
            _dashboardService = dashboardService;
        }

        public async Task AddDataPoint(ExperimentData data)
        {
            await _dashboardService.AddDataPoint(data);
        }

        public async Task AddDataBatch(IList<ExperimentData> batch)
        {
            foreach (var data in batch)
            {
                await _dashboardService.AddDataPoint(data);
            }
        }
    }
}