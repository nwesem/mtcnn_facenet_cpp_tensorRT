#ifndef TRT_L2NORM_HELPER_PLUGIN_H
#define TRT_L2NORM_HELPER_PLUGIN_H
#include "NvInferPlugin.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <vector>

#define ASSERT(assertion)                                        \
    {                                                            \
        if (!(assertion))                                        \
        {                                                        \
            std::cout<<"ASSERTION FAILED in "                    \
                     <<__FILE__<<":"<<__LINE__                   \
                     <<std::endl;                                \
            abort();                                             \
        }                                                        \
    }

using namespace std;
using namespace nvinfer1;
using namespace nvinfer1::plugin;

typedef enum
{
  OP_TYPE_MAX=0,
  OP_TYPE_RSQRT=1,
  OP_TYPE_SQRT=2
} operation_t;

class L2NormHelper : public IPluginV2
{
public:
    L2NormHelper(int op_type, float eps);

    L2NormHelper(int op_type, float eps, int C, int H, int W);

    L2NormHelper(const void* buffer, size_t length);

    ~L2NormHelper() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2* clone() const;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    void configureWithFormat(
        const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        DataType type, PluginFormat format, int maxBatchSize) override;

private:
    int C, H, W;
    int op_type;
    float eps = 1e-12;
    DataType mDataType{DataType::kHALF};
    const char* mPluginNamespace;
};

class L2NormHelperPluginCreator : public IPluginCreator
{
public:
    L2NormHelperPluginCreator();

    ~L2NormHelperPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    void setPluginNamespace(const char* ns) override;

    const char* getPluginNamespace() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    int mOpType;
    float mEps;
    static vector<PluginField> mPluginAttributes;

protected:
    string mNamespace;
};

// Write values into buffer
template <typename T>
void write(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
T read(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

REGISTER_TENSORRT_PLUGIN(L2NormHelperPluginCreator);

#endif // TRT_L2NORM_HELPER_PLUGIN_H
