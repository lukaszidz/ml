using Microsoft.ML.OnnxRuntime;

namespace Onnx.ShuffleNet
{
    internal class Inference
    {
        private string _modelFilePath;

        public Inference(string modelFilePath)
        {
            _modelFilePath = modelFilePath;
        }

        public IEnumerable<float> Run(IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            var inferResults = Infer(inputs);
            IEnumerable<float> output = inferResults.First().AsEnumerable<float>();
            float sum = output.Sum(x => (float)Math.Exp(x));
            return output.Select(x => (float)Math.Exp(x) / sum);
        }

        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Infer(IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            using var session = new InferenceSession(_modelFilePath);
            var results = session.Run(inputs);
            return results;
        }
    }

    internal class InferenceResult
    {
        public string Label { get; set; }
        public float Score { get; set; }
    }
}
