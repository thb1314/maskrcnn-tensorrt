#ifndef TRT_GRID_SAMPLER_HPP
#define TRT_GRID_SAMPLER_HPP
#include <cublas_v2.h>

#include <memory>
#include <string>
#include <vector>

#include "trt_plugin_helper.hpp"

namespace mmcv {
enum class GridSamplerInterpolation { Bilinear, Nearest };
enum class GridSamplerPadding { Zeros, Border, Reflection };
}  // namespace mmcv

class GridSamplerDynamic : public nvinfer1::IPluginV2DynamicExt {
 public:
  GridSamplerDynamic(const std::string &name, int mode, int paddingMode,
                     bool alignCorners);

  GridSamplerDynamic(const std::string name, const void *data, size_t length);

  GridSamplerDynamic() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
      nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc *inOut,
                                 int nbInputs, int nbOutputs) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out,
                       int nbOutputs) noexcept override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs,
                          int nbOutputs) const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
              const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace,
              cudaStream_t stream) noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType *inputTypes,
                                       int nbInputs) const noexcept override;

  // IPluginV2 Methods
  const char *getPluginType() const noexcept override;
  const char *getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char *pluginNamespace) noexcept override;
  const char *getPluginNamespace() const noexcept override;

 private:
  const std::string mLayerName;
  std::string mNamespace;

  int mMode;
  int mPaddingMode;
  bool mAlignCorners;

};

class GridSamplerDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  GridSamplerDynamicCreator();

  const char *getPluginName() const noexcept override;

  const char *getPluginVersion() const noexcept override;

  const nvinfer1::PluginFieldCollection *getFieldNames() noexcept override;

  nvinfer1::IPluginV2 *createPlugin(
      const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept override;

  nvinfer1::IPluginV2 *deserializePlugin(const char *name,
                                         const void *serialData,
                                         size_t serialLength) noexcept override;

  void setPluginNamespace(const char *pluginNamespace) noexcept override;

  const char *getPluginNamespace() const noexcept override;

 private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};
#endif  // TRT_GRID_SAMPLER_HPP
