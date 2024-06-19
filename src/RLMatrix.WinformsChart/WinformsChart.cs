using System.Windows.Forms.DataVisualization.Charting;
using RLMatrix.Agents.Common;

namespace RLMatrix.WinformsChart
{
    public class WinformsChart : RLMatrix.Agents.Common.IRLChartService
    {
        public WinformsChart()
        {
            CreateChartForm();
        }

        public void CreateOrUpdateChart(List<double> episodeRewards)
        {
            if(chartForm == null)
            {
                CreateChartForm();
            }
            
            episodeCounter = episodeRewards.Count;
            WaitChartFormReady();

            this.episodeRewards = episodeRewards;
            UpdateChart();

        }

        #region plot
        Form chartForm;
        Chart chart;
        int episodeCounter;
        private ManualResetEvent chartFormReady = new ManualResetEvent(false);

        private async void CreateChartForm()
        {
            await Task.Run(() =>
            {
                chartForm = new Form();
                chart = new Chart();
                chart.ChartAreas.Add(new ChartArea());
                chartForm.Controls.Add(chart);
                chart.Dock = DockStyle.Fill;

                // Signal that the chartForm is ready
                chartFormReady.Set();

                Application.Run(chartForm);
            });
        }
        private void WaitChartFormReady()
        {
            // Wait until chartForm is ready
            chartFormReady.WaitOne();
        }


        private List<double> episodeRewards = new();
        public void UpdateChart()
        {
            if (chart.InvokeRequired)
            {
                // Invoke the method on the UI threads
                chart.Invoke(new Action(() => UpdateChart()));
            }
            else
            {
                // Clear the chart and add a new series
                chart.Series.Clear();
                var series = new Series();
                series.ChartType = SeriesChartType.Line;
                series.MarkerStyle = MarkerStyle.Circle;
                series.MarkerSize = 8; // Set the marker size to a larger valu

                // Add points to the series
                for (int i = 0; i < episodeRewards.Count; i++)
                {
                    series.Points.AddXY(i, episodeRewards[i]);
                }

                // Add the series to the chart
                chart.Series.Add(series);

                // Set autoscaling for the y-axis
                chart.ChartAreas[0].AxisY.Minimum = episodeRewards?.Any() == true ? Math.Round(episodeRewards.Min()) : 0;
                chart.ChartAreas[0].AxisY.Maximum = episodeRewards?.Any() == true ? Math.Round(episodeRewards.Max()) : 0;
                chart.ChartAreas[0].AxisY.Interval = Math.Round((Math.Abs(chart.ChartAreas[0].AxisY.Maximum) + Math.Abs(chart.ChartAreas[0].AxisY.Minimum)) / 10);

                // Set the x-axis interval based on the number of data points
                chart.ChartAreas[0].AxisX.Interval = 5;
                chart.ChartAreas[0].AxisX.Minimum = 0;
                chart.ChartAreas[0].AxisX.Maximum = episodeCounter;

                // Refresh the chart
                chart.Invalidate();
            }
        }
        #endregion
    }
}
