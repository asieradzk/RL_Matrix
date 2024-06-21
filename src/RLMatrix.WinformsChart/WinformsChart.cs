using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using RLMatrix.Agents.Common;

namespace RLMatrix.WinformsChart
{
    public class WinformsChart : IRLChartService
    {
        public WinformsChart()
        {
            CreateChartForm();
        }

        public void CreateOrUpdateChart(List<double> episodeRewards)
        {
            if (chartForm == null)
            {
                CreateChartForm();
            }

            episodeCounter = episodeRewards.Count;
            WaitChartFormReady();
            this.episodeRewards = episodeRewards;
            UpdateChart();
        }

        Form chartForm;
        Chart chart;
        int episodeCounter;
        private ManualResetEvent chartFormReady = new ManualResetEvent(false);
        private List<double> episodeRewards = new();

        private async void CreateChartForm()
        {
            try
            {
                await Task.Run(() =>
                {
                    chartForm = new Form();
                    chart = new Chart();
                    chart.ChartAreas.Add(new ChartArea());
                    chartForm.Controls.Add(chart);
                    chart.Dock = DockStyle.Fill;
                    chartFormReady.Set();
                    try
                    {
                        Application.Run(chartForm);
                    }
                    catch (Exception)
                    {
                    }
                });
            }
            catch (System.Exception)
            {
            }
        }

        private void WaitChartFormReady()
        {
            chartFormReady.WaitOne();
        }

        public void UpdateChart()
        {
            if (chart.InvokeRequired)
            {
                chart.Invoke(new Action(() => UpdateChart()));
            }
            else
            {
                chart.Series.Clear();
                var series = new Series
                {
                    ChartType = SeriesChartType.Line,
                    MarkerStyle = MarkerStyle.Circle,
                    MarkerSize = 8
                };

                for (int i = 0; i < episodeRewards.Count; i++)
                {
                    series.Points.AddXY(i, episodeRewards[i]);
                }

                chart.Series.Add(series);

                double minY = episodeRewards.Any() ? episodeRewards.Min() : 0;
                double maxY = episodeRewards.Any() ? episodeRewards.Max() : 1;

                if (Math.Abs(maxY - minY) < 0.001)
                {
                    minY -= 0.5;
                    maxY += 0.5;
                }

                chart.ChartAreas[0].AxisY.Minimum = Math.Floor(minY);
                chart.ChartAreas[0].AxisY.Maximum = Math.Ceiling(maxY);

                double range = chart.ChartAreas[0].AxisY.Maximum - chart.ChartAreas[0].AxisY.Minimum;
                chart.ChartAreas[0].AxisY.Interval = CalculateNiceInterval(range);

                chart.ChartAreas[0].AxisX.Minimum = 0;
                chart.ChartAreas[0].AxisX.Maximum = Math.Max(episodeCounter, 10);
                chart.ChartAreas[0].AxisX.Interval = Math.Max(1, Math.Floor(episodeCounter / 10.0));

                chart.Invalidate();
            }
        }

        private double CalculateNiceInterval(double range)
        {
            double exponent = Math.Floor(Math.Log10(range));
            double fraction = range / Math.Pow(10, exponent);

            double niceFraction;
            if (fraction < 1.5)
                niceFraction = 1;
            else if (fraction < 3)
                niceFraction = 2;
            else if (fraction < 7)
                niceFraction = 5;
            else
                niceFraction = 10;

            return niceFraction * Math.Pow(10, exponent - 1);
        }
    }
}