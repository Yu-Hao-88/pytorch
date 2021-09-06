// 引入PyTorch 程式的標頭檔（header）、以及支援JPEG 的CImg
#include "torch/script.h" // <1>
#define cimg_use_jpeg
#include "CImg.h"

using namespace cimg_library;
int main(int argc, char **argv) {

  if (argc != 4) {
    std::cerr << "Call as " << argv[0] << " model.pt input.jpg output.jpg"
              << std::endl;
    return 1;
  }

  CImg<float> image(argv[2]); //匯入圖片，並解碼成浮點數陣列
  image = image.resize(227, 227); //縮小圖片

  auto input_ = torch::tensor(
    torch::ArrayRef<float>(image.data(), image.size())); // 將圖片資料轉換成張量
  auto input = input_.reshape({1, 3, image.height(),
			       image.width()}).div_(255); // 為了讓資料順利從CImg 過渡到PyTorch，需重新調整其形狀和大小

  // 從檔案中匯入JIT 模型或函式
  auto module = torch::jit::load(argv[1]);

  // 將輸入包裝成由IValue（一種能儲存任何值的通用型別）資料所組成的（單元素）向量
  std::vector<torch::jit::IValue> inputs;

  inputs.push_back(input);
  //呼叫模組並產生結果張量。為求效率，我們轉讓了所有權（ownership）；
  //因此，若我們繼續使用整數變數，則該張量之後會清空
  auto output_ = module.forward(inputs).toTensor();

  auto output = output_.contiguous().mul_(255); // 確保結果是連續的
  
  // data_ptr<float>() 方法能傳回張量在記憶體中的位置指標。
  //有了該指標以及形狀資訊後，我們就能建立輸出圖片了
  CImg<float> out_img(output.data_ptr<float>(), output.size(2),
                      output.size(3), 1, output.size(1));
  out_img.save(argv[3]); //儲存圖片
  return 0;
}
