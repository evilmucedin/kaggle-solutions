#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "fileIO.h"
#include "cmdline.h"
#include "csv.h"
#include "exceptions.h"
#include "str.h"
#include "random.h"
#include "math.h"
#include "picture.h"
#include "neural.h"

using namespace std;
using namespace testing;

TEST(Split, Basics)
{
    TStringVector sv;
    Split("a,b,c", ',', &sv);
    EXPECT_EQ(sv.size(), 3);
    EXPECT_EQ(sv[0], "a");
    EXPECT_EQ(sv[1], "b");
    EXPECT_EQ(sv[2], "c");
    EXPECT_EQ(sv[2].length(), 1);
}

TEST(NeuralNet, XOR)
{
    TNeuralEstimator estimator;
    {
        estimator.SetInputSize(2);
        size_t prevLayerBegin = 0;
        size_t prevLayerSize = 2;
        for (size_t iLayer = 0; iLayer < 1; ++iLayer)
        {
            size_t layerBegin = estimator.Size();
            for (size_t i = 0; i < 3; ++i)
            {
                TNeuralEstimator::TNeuron neuron;
                for (size_t j = 0; j < prevLayerSize; ++j)
                {
                    neuron.AddSinapse(prevLayerBegin + j);
                }
                estimator.Add(neuron);
            }
            prevLayerBegin = layerBegin;
            prevLayerSize = 3;
        }
        TNeuralEstimator::TNeuron neuronOutput;
        for (size_t j = 0; j < prevLayerSize; ++j)
        {
            neuronOutput.AddSinapse(prevLayerBegin + j);
        }
        estimator.Add(neuronOutput);
        estimator.Prepare();
    }

    {
        for (size_t iLearnIt = 0; iLearnIt < 100000; ++iLearnIt)
        {
            {
                estimator.Inflate();
                for (size_t x = 0; x < 2; ++x)
                {
                    for (size_t y = 0; y < 2; ++y)
                    {
                        TFloatVector input;
                        input.push_back(x);
                        input.push_back(y);
                        const float result = ((x ^ y) == 0) ? 0.f : 1.f;
                        estimator.BackPropagation(input, result, iLearnIt);
                    }
                }
            }
            {
                float error = 0.f;
                for (size_t x = 0; x < 2; ++x)
                {
                    for (size_t y = 0; y < 2; ++y)
                    {
                        TFloatVector input;
                        input.push_back(x);
                        input.push_back(y);
                        const float result = ((x ^ y) == 0) ? 0.f : 1.f;
                        const float netResult = estimator.GetOutput(input);
                        // printf("%d %d %f %f\n", (int)x, (int)y, result, netResult);
                        error += Sqr(result - netResult);
                    }
                }
                // printf("Error %d: %f\n", iLearnIt, error);
            }
        }
    }
}

TEST(NeuralNet, SaveLoad)
{
    TNeuralEstimator estimator;
    {
        estimator.SetInputSize(2);
        size_t prevLayerBegin = 0;
        size_t prevLayerSize = 2;
        for (size_t iLayer = 0; iLayer < 1; ++iLayer)
        {
            size_t layerBegin = estimator.Size();
            for (size_t i = 0; i < 3; ++i)
            {
                TNeuralEstimator::TNeuron neuron;
                for (size_t j = 0; j < prevLayerSize; ++j)
                {
                    neuron.AddSinapse(prevLayerBegin + j);
                }
                estimator.Add(neuron);
            }
            prevLayerBegin = layerBegin;
            prevLayerSize = 3;
        }
        TNeuralEstimator::TNeuron neuronOutput;
        for (size_t j = 0; j < prevLayerSize; ++j)
        {
            neuronOutput.AddSinapse(prevLayerBegin + j);
        }
        estimator.Add(neuronOutput);
        estimator.Prepare();
    }

    {
        for (size_t iLearnIt = 0; iLearnIt < 10; ++iLearnIt)
        {
            {
                estimator.Inflate();
                for (size_t x = 0; x < 2; ++x)
                {
                    for (size_t y = 0; y < 2; ++y)
                    {
                        TFloatVector input;
                        input.push_back(x);
                        input.push_back(y);
                        const float result = ((x ^ y) == 0) ? 0.f : 1.f;
                        estimator.BackPropagation(input, result, iLearnIt);
                    }
                }
            }
            {
                float error = 0.f;
                for (size_t x = 0; x < 2; ++x)
                {
                    for (size_t y = 0; y < 2; ++y)
                    {
                        TFloatVector input;
                        input.push_back(x);
                        input.push_back(y);
                        const float result = ((x ^ y) == 0) ? 0.f : 1.f;
                        const float netResult = estimator.GetOutput(input);
                        error += Sqr(result - netResult);
                    }
                }
            }
        }
    }

    {
        float error = 0.f;
        for (size_t x = 0; x < 2; ++x)
        {
            for (size_t y = 0; y < 2; ++y)
            {
                TFloatVector input;
                input.push_back(x);
                input.push_back(y);
                const float result = ((x ^ y) == 0) ? 0.f : 1.f;
                const float netResult = estimator.GetOutput(input);
                error += Sqr(result - netResult);
            }
        }
        estimator.Save("test.weights");
        estimator.Load("test.weights");
        for (size_t x = 0; x < 2; ++x)
        {
            for (size_t y = 0; y < 2; ++y)
            {
                TFloatVector input;
                input.push_back(x);
                input.push_back(y);
                const float result = ((x ^ y) == 0) ? 0.f : 1.f;
                const float netResult = estimator.GetOutput(input);
                error -= Sqr(result - netResult);
            }
        }
        EXPECT_TRUE(fabs(error) < 0.000001f);
    }
}

typedef vector<float> TFloatVector;
typedef vector<TFloatVector> TFloatMatrix;

struct TCosineEstimator : public IProbEstimator
{
    struct TWeights : public TFloatMatrix
    {
        TWeights()
        {
            Init();
        }

        TWeights(const TPicture& picture)
        {
            Init();
            for (size_t i = 0; i < TPicture::SIZE; ++i)
            {
                for (size_t j = 0; j < TPicture::SIZE; ++j)
                {
                    (*this)[i][j] = picture.Get(i, j);
                }
            }
        }

        void Init()
        {
            TFloatVector dummy(TPicture::SIZE);
            resize(TPicture::SIZE, dummy);
        }

        void Add(const TPicture& picture)
        {
            for (size_t i = 0; i < TPicture::SIZE; ++i)
            {
                for (size_t j = 0; j < TPicture::SIZE; ++j)
                {
                    (*this)[i][j] += picture.Get(i, j);
                }
            }
        }

        void Normalize()
        {
            float normalizer = 0.f;
            for (size_t i = 0; i < TPicture::SIZE; ++i)
            {
                for (size_t j = 0; j < TPicture::SIZE; ++j)
                {
                    normalizer += Sqr((*this)[i][j]);
                }
            }
            normalizer = sqrtf(normalizer);
            for (size_t i = 0; i < TPicture::SIZE; ++i)
            {
                for (size_t j = 0; j < TPicture::SIZE; ++j)
                {
                    (*this)[i][j] /= normalizer;
                }
            }
        }

        float Mul(const TWeights& w) const
        {
            float result = 0.f;
            for (size_t i = 0; i < TPicture::SIZE; ++i)
            {
                for (size_t j = 0; j < TPicture::SIZE; ++j)
                {
                    result += (*this)[i][j]*w[i][j];
                }
            }
            return result;
        }
    };

    TWeights m_weights;

    TCosineEstimator()
    {
    }

    void Learn(const TPicture& picture)
    {
        m_weights.Add(picture);
    }

    void EndLearn()
    {
        m_weights.Normalize();
    }

    virtual float Estimate(const TPicture& picture) const
    {
        TWeights pictureWeights(picture);
        pictureWeights.Normalize();
        return m_weights.Mul(pictureWeights);
    }
};

void SplitLearnTest(const TRows& input, float ratio, TRows* learn, TRows* test)
{
    TTimer timer("SplitTestLearn");
    learn->clear();
    test->clear();
    for (TRows::const_iterator toRow = input.begin(); toRow != input.end(); ++toRow)
    {
        if (Rand01() < ratio)
        {
            learn->push_back(*toRow);
        }
        else
        {
            test->push_back(*toRow);
        }
    }
}

typedef pair<size_t, float> TBest;

TBest Choose(IProbEstimators estimators, const TPicture& picture, const string& name, size_t index)
{
    float best = 0.f;
    float nextToBest = 0.f;
    size_t bestIndex = 0;
    vector<float> probes(estimators.size());
    for (size_t i = 0; i < estimators.size(); ++i)
    {
        const float prob = estimators[i]->Estimate(picture);
        probes[i] = prob;
        if (prob > best)
        {
            nextToBest = best;
            best = prob;
            bestIndex = i;
        }
        else
        {
            nextToBest = std::max(prob, nextToBest);
        }
    }
    if ((best < 0.8f) || (nextToBest + 0.2f > best))
    {
        MkDir(name);
        picture.SaveBMP(name + "/" + std::to_string(index) + ".bmp");
        TFileWriter fOut(name + "/" + std::to_string(index) + ".txt");
        fOut.Write(std::to_string(bestIndex) + "\t" + std::to_string(best) + "\t" + std::to_string(nextToBest) + "\n");
        for (size_t i = 0; i < probes.size(); ++i)
        {
            fOut.Write( std::to_string(i) + "\t" + std::to_string(probes[i]) + "\n" );
        }
        picture.Write(fOut);
    }
    return make_pair(bestIndex, best);
}

void MakePictures(const TRows& rows, TPictures* pictures, bool test)
{
    TTimer timerLearn("MakePictures " + std::to_string(rows.size()));
    pictures->clear();
    for (size_t i = 0; i < rows.size(); ++i)
    {
        pictures->push_back( TPicture(rows[i], test) );
    }
}

int main(int argc, char* argv[])
{
    TCommandLineParser parser(argc, argv);
    const bool unittests = parser.Has('u', "unittests", "run unittests");
    const bool draw = parser.Has('d', "draw", "draw train");
    const bool verbose = parser.Has('v', "verbose", "verbose");
    const bool cosine = parser.Has('c', "cosine", "cosine");
    const bool neural = parser.Has('n', "neural", "neural");
    const int limit = parser.GetInt('l', "limit", "limit input", std::numeric_limits<int>::max());
    const string loadFrom = parser.Get('L', "load", "start from NN", "");
    parser.AutoUsage();

    if (draw)
    {
        TCSVReader trainData("train.csv", true);
        for (size_t i = 0; i < trainData.m_rows.size(); ++i)
        {
            TPicture(trainData.m_rows[i], false).Draw();
            printf("\n");
        }
    }
    else if (unittests)
    {
        InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
    else if (cosine)
    {
        TCSVReader trainData("train.csv", true);
        {
            TRows learn;
            TRows test;
            SplitLearnTest(trainData.m_rows, 0.9f, &learn, &test);

            TCosineEstimator estimators[10];
            {
                TTimer timerLearn("Learn");
                for (size_t i = 0; i < learn.size(); ++i)
                {
                    TPicture p(learn[i], false);
                    // p.Crop();
                    estimators[learn[i][0]].Learn(p);
                }
            }

            {
                IProbEstimators pEstimators(10);
                for (size_t i = 0; i < 10; ++i)
                {
                    pEstimators[i] = &estimators[i];
                }

                TTimer timerLearn("Test");
                size_t preceision = 0;
                for (size_t i = 0; i < test.size(); ++i)
                {
                    TPicture p(test[i], false);
                    // p.Crop();
                    TBest best = Choose(pEstimators, p, "test", i);
                    if (verbose)
                    {
                        printf("%d %d\n", (int)test[i][0], best.first);
                        p.Draw();
                    }
                    if (test[i][0] == best.first)
                    {
                        ++preceision;
                    }
                }
                printf("Precision: %f\n", ((float)preceision)/test.size());
            }
        }

        {
            TCosineEstimator estimators[10];
            {
                TTimer timerLearn("Learn");
                for (size_t i = 0; i < trainData.m_rows.size(); ++i)
                {
                    TPicture p(trainData.m_rows[i], false);
                    estimators[trainData.m_rows[i][0]].Learn(p);
                }
            }

            {
                IProbEstimators pEstimators(10);
                for (size_t i = 0; i < 10; ++i)
                {
                    pEstimators[i] = &estimators[i];
                }

                TTimer timerLearn("Test");
                TCSVReader testData("test.csv", true);
                TCSVWriter writer("cosine.csv");
                for (size_t i = 0; i < testData.m_rows.size(); ++i)
                {
                    TPicture p(testData.m_rows[i], true);
                    TBest best = Choose(pEstimators, p, "test", i);
                    writer.Put( std::to_string(best.first) );
                    writer.NewLine();
                    if (verbose)
                    {
                        printf("%d\n", best.first);
                        p.Draw();
                    }
                }
            }
        }
    }
    else if (neural)
    {
        TPictures pLearn;
        {
            TCSVReader trainData("train.csv", true, limit);
            MakePictures(trainData.m_rows, &pLearn, false);
        }
        TPictures pTest;
        {
            TCSVReader testData("test.csv", true);
            MakePictures(testData.m_rows, &pTest, true);
        }
        {
            TPictures pHand;
            {
                TCSVReader handData("hand.csv", true);
                MakePictures(handData.m_rows, &pHand, false);
            }
            pLearn.insert(pLearn.end(), pHand.begin(), pHand.end());
        }
        {
            TCosineEstimator estimators[10];
            IProbEstimators pEstimators(10);
            {
                TTimer timerCosine("Learn cosine");
                {
                    for (size_t i = 0; i < pLearn.size(); ++i)
                    {
                        estimators[pLearn[i].Digit()].Learn(pLearn[i]);
                    }
                }
            }
            for (size_t i = 0; i < 10; ++i)
            {
                estimators[i].EndLearn();
                pEstimators[i] = &estimators[i];
            }
            {
                TTimer timerCosine("Update features");
                for (size_t i = 0; i < pLearn.size(); ++i)
                {
                    pLearn[i].CalcFeaturesPhase2(pEstimators);
                }
                for (size_t i = 0; i < pTest.size(); ++i)
                {
                    pTest[i].CalcFeaturesPhase2(pEstimators);
                }
            }
        }
        printf("Train=%d, Test=%d\n", static_cast<int>(pLearn.size()), static_cast<int>(pTest.size()));
        vector<TNeuralEstimator> estimators(10);
        {
            TTimer timerLearn("Configure neural net");
            for (size_t digit = 0; digit < 10; ++digit)
            {
                TNeuralEstimator& estimator = estimators[digit];

                estimator.SetInputSize(TPicture::VECTOR_SIZE);
                size_t prevLayerBegin = 1;
                size_t prevLayerSize = TPicture::VECTOR_SIZE - 1;
                for (size_t i = 0; i <= 2; ++i)
                {
                    const size_t layerBegin = estimator.Size();
                    const size_t layerSize = (i == 2) ? 1 : ( (i == 0) ? TPicture::VECTOR_SIZE : 100 );
                    for (size_t k = 0; k < layerSize; ++k)
                    {
                        TNeuralEstimator::TNeuron neuron;
                        for (size_t j = 0; j < prevLayerSize; ++j)
                        {
                            neuron.AddSinapse(prevLayerBegin + j, ( (k + j + i) % TPicture::VECTOR_SIZE ) ? 0.f : 1.f);
                        }
                        neuron.AddSinapse(0, 0.000001f);
                        estimator.Add(neuron);
                    }
                    prevLayerBegin = layerBegin;
                    prevLayerSize = layerSize;
                }
                estimator.Prepare();
                if (!loadFrom.empty())
                {
                    estimator.Load(loadFrom + std::to_string(digit));
                }
            }
        }

        {
            TTimer timerLearn("Learn");
            for (size_t iLearnIt = 0; iLearnIt < 100; ++iLearnIt)
            {
                float ratio;
                switch (iLearnIt)
                {
                    case 0:
                        ratio = 0.01f;
                        break;
                    case 1:
                        ratio = 0.05f;
                        break;
                    case 2:
                        ratio = 0.2f;
                        break;
                    case 3:
                        ratio = 0.5f;
                        break;
                    default:
                        ratio = 1.1f;
                }
                const string name = "debug/learn" + std::to_string(iLearnIt);
                MkDir(name);
                {
                    TTimer timerLearn("Learn iteration " + std::to_string(iLearnIt) + " " + std::to_string(ratio));
                    for (size_t digit = 0; digit < 10; ++digit)
                    {
                        TNeuralEstimator& estimator = estimators[digit];
                        TTimer timerLearn2("Learn iteration " + std::to_string(iLearnIt) + " digit " + std::to_string(digit));
                        estimator.Inflate();

                        vector<size_t> indexes(pLearn.size());
                        for (size_t i = 0; i < indexes.size(); ++i)
                        {
                            indexes[i] = i;
                        }
                        Shuffle(indexes);

                        for (size_t i = 0; i < pLearn.size(); ++i)
                        {
                            if (Rand01() > ratio)
                            {
                                continue;
                            }

                            const TPicture& p = pLearn[indexes[i]];
                            const float value = (p.Digit() == digit) ? 1.f : 0.f;
                            if (!(i % 100))
                            {
                                printf("before %f %f %f\n", (float)i/pLearn.size(), value, estimator.GetOutput(p.AsVector()));
                            }
                            estimator.BackPropagation(p.AsVector(), value, iLearnIt);
                            if (!(i % 100))
                            {
                                printf("after %f %f %f\n", (float)i/pLearn.size(), value, estimator.GetOutput(p.AsVector()));
                                fflush(stdout);
                            }
                        }
                        estimator.Save(name + "/NN" + std::to_string(digit));
                    }
                }
                {
                    TTimer timerTest("Test iteration " + std::to_string(iLearnIt));
                    for (size_t digit = 0; digit < 10; ++digit)
                    {
                        float error = 0.f;
                        size_t num = 0;
                        for (size_t i = 0; i < pLearn.size(); ++i)
                        {
                            if (Rand01() > 2.f*ratio)
                            {
                                continue;
                            }

                            const TPicture& p = pLearn[i];

                            if (i && !(i % 4000))
                            {
                                printf("%.2f\n", ((float)i)/pLearn.size());
                                fflush(stdout);
                            }

                            const float result = (p.Digit() == digit) ? 1.f : 0.f;
                            const float netResult = estimators[digit].GetOutput(p.AsVector());
                            const float dError = Sqr(result - netResult);
                            if (dError > 0.25f)
                            {
                                p.SaveBMP(name + "/" + std::to_string(i) + ".bmp");
                                TFileWriter fOut(name + "/" + std::to_string(i) + "_" + std::to_string(digit) + ".txt");
                                fOut.Write( std::to_string(i) + "\t" + std::to_string(p.Digit()) + "\t" + std::to_string(digit) + "\t" + std::to_string(result) + "\t" + std::to_string(netResult) + "\n" );
                                p.Write(fOut);
                            }
                            error += dError;
                            ++num;
                        }
                        printf("Error %d %d: %f %f\n", (int)iLearnIt, (int)digit, error, error/num);
                    }
                }
                const string testName = "debug/testNN" + std::to_string(iLearnIt);
                {
                    TTimer timerApply("Apply " + std::to_string(iLearnIt));
                    TCSVWriter writer("neural.csv");
                    writer.Put("ImageId");
                    writer.Put("Label");
                    writer.NewLine();
                    IProbEstimators pEstimators(10);
                    for (size_t i = 0; i < 10; ++i)
                    {
                        pEstimators[i] = &estimators[i];
                    }

                    TFileWriter fOut("dump.txt");
                    for (size_t i = 0; i < pTest.size(); ++i)
                    {
                        if (!(i % 500))
                        {
                            printf("%f\n", ((float)i)/pTest.size());
                            fflush(stdout);
                        }

                        const TPicture& p = pTest[i];
                        TBest best = Choose(pEstimators, p, testName, i);
                        writer.Put( std::to_string(i + 1) );
                        writer.Put( std::to_string(best.first) );
                        writer.NewLine();
                        if (verbose)
                        {
                            printf("%d\n", best.first);
                            p.Draw();
                        }

                        for (size_t j = 0; j < 10; ++j)
                        {
                            fOut.Write(std::to_string(i) + "\t" + std::to_string(j) + "\t" + std::to_string(estimators[j].Estimate(p)) + "\n");
                        }
                        fOut.Write(std::to_string(i) + "\t" + std::to_string(best.first) + "\t" + std::to_string(best.second) + "\n");
                        p.Draw(fOut.GetHandle());
                        p.DrawLine(fOut.GetHandle());
                    }
                }
            }
        }
    }
    else
    {
        TPictures pLearn;
        TPictures pTest;
        {
            TCSVReader trainData("train.csv", true, limit);
            TRows learn;
            TRows test;
            SplitLearnTest(trainData.m_rows, 0.9f, &learn, &test);
            MakePictures(learn, &pLearn, false);
            MakePictures(test, &pTest, false);
        }

        TNeuralEstimator estimator;
        {
            TTimer timerLearn("Configure neural net");
            estimator.SetInputSize(TPicture::VECTOR_SIZE);
            size_t prevLayerBegin = 0;
            size_t prevLayerSize = TPicture::VECTOR_SIZE;
            for (size_t i = 0; i < 2; ++i)
            {
                const size_t layerBegin = estimator.Size();
                const size_t layerSize = (i == 0) ? TPicture::VECTOR_SIZE : 100;
                for (size_t k = 0; k < layerSize; ++k)
                {
                    TNeuralEstimator::TNeuron neuron;
                    for (size_t j = 0; j < prevLayerSize; ++j)
                    {
                        neuron.AddSinapse(prevLayerBegin + j);
                    }
                    estimator.Add(neuron);
                }
                prevLayerBegin = layerBegin;
                prevLayerSize = layerSize;
            }
            {
                TNeuralEstimator::TNeuron neuronOutput;
                for (size_t j = 0; j < prevLayerSize; ++j)
                {
                    neuronOutput.AddSinapse(prevLayerBegin + j);
                }
                estimator.Add(neuronOutput);
            }
            estimator.Prepare();
        }

        {
            TTimer timerLearn("Learn " + std::to_string(pLearn.size()));
            for (size_t iLearnIt = 0; iLearnIt < 100; ++iLearnIt)
            {
                {
                    TTimer timerLearn("Learn iteration " + std::to_string(iLearnIt));
                    estimator.Inflate();
                    for (size_t i = 0; i < pLearn.size(); ++i)
                    {
                        const TPicture& p = pLearn[i];
                        const float value = (p.Digit() == 0) ? 1.f : 0.f;
                        if (!(i % 100))
                        {
                            printf("before %f %f %f\n", (float)i/pLearn.size(), value, estimator.GetOutput(p.AsVector()));
                        }
                        estimator.BackPropagation(p.AsVector(), value, iLearnIt);
                        if (!(i % 100))
                        {
                            printf("after %f %f %f\n", (float)i/pLearn.size(), value, estimator.GetOutput(p.AsVector()));
                        }
                    }
                }
                {
                    TTimer timerLearn("Test iteration " + std::to_string(iLearnIt));
                    float error = 0.f;
                    for (size_t i = 0; i < pTest.size(); ++i)
                    {
                        const TPicture& p = pTest[i];
                        const float result = (p.Digit() == 0) ? 1.f : 0.f;
                        const float netResult = estimator.GetOutput(p.AsVector());
                        error += Sqr(result - netResult);
                    }
                    printf("Error %d: %f %f\n", iLearnIt, error, error/pTest.size());
                }
            }
        }
    }

    return 0;
}
