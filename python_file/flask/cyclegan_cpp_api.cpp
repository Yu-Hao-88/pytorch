// 一次性匯入torch/torch.h 標頭檔與CImg
#include <torch/torch.h>
#define cimg_use_jpeg
#include <CImg.h>
using torch::Tensor;  // 一直打『torch::Tensor』很麻煩，故這裡將其加入主要命名空間中

// at the time of writing this code (shortly after PyTorch 1.3),
// the C++ api wasn't complete and (in the case of ReLU) bug-free,
// so we define some Modules ad-hoc here.
// Chances are, that you can take standard models if and when
// they are done.

struct ConvTranspose2d : torch::nn::Module {
  // we don't do any of the running stats business
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> output_padding_;
  std::vector<int64_t> dilation_;
  Tensor weight;
  Tensor bias;

  ConvTranspose2d(int64_t in_channels, int64_t out_channels,
                  int64_t kernel_size, int64_t stride, int64_t padding,
                  int64_t output_padding)
      : stride_(2, stride), padding_(2, padding),
        output_padding_(2, output_padding), dilation_(2, 1) {
    // not good init...
    weight = register_parameter(
        "weight",
        torch::randn({out_channels, in_channels, kernel_size, kernel_size}));
    bias = register_parameter("bias", torch::randn({out_channels}));
  }
  Tensor forward(const Tensor &inp) {
    return conv_transpose2d(inp, weight, bias, stride_, padding_,
                            output_padding_, /*groups=*/1, dilation_);
  }
};

// tag::block[]
struct ResNetBlock : torch::nn::Module {
  torch::nn::Sequential conv_block;
  ResNetBlock(int64_t dim)
      : conv_block(  // 初始化Sequential，將所有子模組包含進來
            torch::nn::ReflectionPad2d(1),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 3)),
            torch::nn::InstanceNorm2d(
	       torch::nn::InstanceNorm2dOptions(dim)),
            torch::nn::ReLU(/*inplace=*/true),
	    torch::nn::ReflectionPad2d(1),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 3)),
            torch::nn::InstanceNorm2d(
	       torch::nn::InstanceNorm2dOptions(dim))) {
    register_module("conv_block", conv_block); // 請務必記得登錄你所指定的模組，否則會出錯！
  }

  Tensor forward(const Tensor &inp) {
    return inp + conv_block->forward(inp); // <3>
  }
};
// end::block[]

// tag::generator1[]
struct ResNetGeneratorImpl : torch::nn::Module {
  torch::nn::Sequential model;
  ResNetGeneratorImpl(int64_t input_nc = 3, int64_t output_nc = 3,
                      int64_t ngf = 64, int64_t n_blocks = 9) {
    TORCH_CHECK(n_blocks >= 0);
    //將模組加至Sequential 容器內，這允許我們把不定數量的模組加入for 迴圈中
    model->push_back(torch::nn::ReflectionPad2d(3)); // <1>
// end::generator1[]
    model->push_back(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(input_nc, ngf, 7)));
    model->push_back(
        torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(7)));
    model->push_back(torch::nn::ReLU(/*inplace=*/true));
    constexpr int64_t n_downsampling = 2;

    for (int64_t i = 0; i < n_downsampling; i++) {
      int64_t mult = 1 << i;
      // tag::generator2[]
      model->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(ngf * mult, ngf * mult * 2, 3)
              .stride(2)
              .padding(1))); // Options 的實際使用範例
      // end::generator2[]
      model->push_back(torch::nn::InstanceNorm2d(
          torch::nn::InstanceNorm2dOptions(ngf * mult * 2)));
      model->push_back(torch::nn::ReLU(/*inplace=*/true));
    }

    int64_t mult = 1 << n_downsampling;
    for (int64_t i = 0; i < n_blocks; i++) {
      model->push_back(ResNetBlock(ngf * mult));
    }
    for (int64_t i = 0; i < n_downsampling; i++) {
      int64_t mult = 1 << (n_downsampling - i);
      model->push_back(
          ConvTranspose2d(ngf * mult, ngf * mult / 2, /*kernel_size=*/3,
                          /*stride=*/2, /*padding=*/1, /*output_padding=*/1));
      model->push_back(torch::nn::InstanceNorm2d(
          torch::nn::InstanceNorm2dOptions((ngf * mult / 2))));
      model->push_back(torch::nn::ReLU(/*inplace=*/true));
    }
    model->push_back(torch::nn::ReflectionPad2d(3));
    model->push_back(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(ngf, output_nc, 7)));
    model->push_back(torch::nn::Tanh());
    // tag::generator3[]
    register_module("model", model);
  }
  Tensor forward(const Tensor &inp) { return model->forward(inp); }
};

// 在ResNetGeneratorImpl 類別中建立一個ResNetGenerator 包裹器
TORCH_MODULE(ResNetGenerator);
// end::generator3[]

int main(int argc, char **argv) {
  // tag::main1[]
  ResNetGenerator model; // 產生模型物件
  // end::main1[]
  if (argc != 3) {
    std::cerr << "call as " << argv[0] << " model_weights.pt image.jpg"
              << std::endl;
    return 1;
  }
  // tag::main2[]
  torch::load(model, argv[1]); // 匯入模型參數
  // end::main2[]
  // you can print the model structure just like you would in PyTorch
  // std::cout << model << std::endl;
  // tag::main3[]
  cimg_library::CImg<float> image(argv[2]);
  image.resize(400, 400);
  auto input_ =
      torch::tensor(torch::ArrayRef<float>(image.data(), image.size()));
  auto input = input_.reshape({1, 3, image.height(), image.width()});

  // 宣告guard 變數的作用與torch.no_grad() 區塊相同；
  //必要時可以將之放在{...} 中來標明其效力（關閉梯度）的範圍
  torch::NoGradGuard no_grad;
  
  model->eval(); //和在Python 中一樣，開啟eval 模式（但就此例的模型而言，此步驟並不重要）
  
  auto output = model->forward(input); //再次強調，這裡要呼叫的是forward，而非模型
  // end::main3[]
  // tag::main4[]
  cimg_library::CImg<float> out_img(output.data_ptr<float>(),
				    output.size(3), output.size(2),
				    1, output.size(1));
  // 顯示圖片（此處不能馬上離開程式，而必須等待使用者按任意鍵關閉圖片視窗）
  cimg_library::CImgDisplay disp(out_img, "See a C++ API zebra!"); // <6>
  while (!disp.is_closed()) {
    disp.wait();
  }
  // end::main4[]
  return 0;
}