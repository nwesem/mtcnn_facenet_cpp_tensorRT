#include <cstring>
#include <iostream>
#include <sstream>

#include "l2norm_helper.h"

using namespace std;
using namespace nvinfer1;

namespace
{
const char* L2NORM_HELPER_PLUGIN_VERSION{"1"};
const char* L2NORM_HELPER_PLUGIN_NAME{"L2Norm_Helper_TRT"};
} // namespace

PluginFieldCollection L2NormHelperPluginCreator::mFC{};
vector<PluginField> L2NormHelperPluginCreator::mPluginAttributes;

L2NormHelper::L2NormHelper(int op_type, float eps): op_type(op_type), eps(eps) {}

L2NormHelper::L2NormHelper(int op_type, float eps, int C, int H, int W):
  op_type(op_type), eps(eps), C(C), H(H), W(W) {}

L2NormHelper::L2NormHelper(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char*>(buffer), *a = d;
    op_type  = read<int>(d);
    eps = read<float>(d);
    C = read<int>(d);
    H = read<int>(d);
    W = read<int>(d);
    mDataType = read<DataType>(d);
    ASSERT(d == a + length);
}

int L2NormHelper::getNbOutputs() const
{
    // Plugin layer has 1 output
    return 1;
}

Dims L2NormHelper::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    ASSERT(inputs[0].nbDims == 3);
    return Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

int L2NormHelper::initialize()
{
    return 0;
}

void L2NormHelper::terminate() {}

size_t L2NormHelper::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

size_t L2NormHelper::getSerializationSize() const
{
    // C, H, W, eps, op_type, mDataType
    return sizeof(int) * 3 + sizeof(float) + sizeof(int) + sizeof(mDataType);
}

void L2NormHelper::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, op_type);
    write(d, eps);
    write(d, C);
    write(d, H);
    write(d, W);
    write(d, mDataType);

    ASSERT(d == a + getSerializationSize());
}

bool L2NormHelper::supportsFormat(DataType type, PluginFormat format) const
{
    return ((type == DataType::kFLOAT || type == DataType::kHALF) &&
            (format == PluginFormat::kNCHW));
}

// Set plugin namespace
void L2NormHelper::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* L2NormHelper::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Configure the layer with input and output data types.
void L2NormHelper::configureWithFormat(
    const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    DataType type, PluginFormat format, int maxBatchSize)
{
    ASSERT(format == PluginFormat::kNCHW);
    ASSERT(type == DataType::kFLOAT || type == DataType::kHALF);
    mDataType = type;
    C = inputDims[0].d[0];
    H = inputDims[0].d[1];
    W = inputDims[0].d[2];
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);
    ASSERT(inputDims[0].nbDims >= 1); // number of dimensions of the input tensor must be >=1
    ASSERT(inputDims[0].d[0] == outputDims[0].d[0] &&
           inputDims[0].d[1] == outputDims[0].d[1] &&
           inputDims[0].d[2] == outputDims[0].d[2]);
}

const char* L2NormHelper::getPluginType() const
{
    return L2NORM_HELPER_PLUGIN_NAME;
}

const char* L2NormHelper::getPluginVersion() const
{
    return L2NORM_HELPER_PLUGIN_VERSION;
}

void L2NormHelper::destroy()
{
    delete this;
}

// Clone the plugin
IPluginV2* L2NormHelper::clone() const
{
    // Create a new instance
    IPluginV2* plugin = new L2NormHelper(op_type, eps);

    // Set the namespace
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

// PluginCreator
L2NormHelperPluginCreator::L2NormHelperPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("op_type", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* L2NormHelperPluginCreator::getPluginName() const
{
    return L2NORM_HELPER_PLUGIN_NAME;
}

const char* L2NormHelperPluginCreator::getPluginVersion() const
{
    return L2NORM_HELPER_PLUGIN_VERSION;
}

void L2NormHelperPluginCreator::setPluginNamespace(const char* ns)
{
    mNamespace = ns;
}

const char* L2NormHelperPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

const PluginFieldCollection* L2NormHelperPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* L2NormHelperPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "op_type"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mOpType = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        if (!strcmp(attrName, "eps"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            mEps = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
    }
    L2NormHelper* obj = new L2NormHelper(mOpType, mEps);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2* L2NormHelperPluginCreator::deserializePlugin(
  const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call L2NormHelper::destroy()
    L2NormHelper* obj = new L2NormHelper(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
