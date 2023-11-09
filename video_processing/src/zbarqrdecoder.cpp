#include "zbarqrdecoder.h"

#include <zbar.h>

using namespace std;
using namespace zbar;

std::vector<std::string> ZBarQRDecoder::Decode(const cv::Mat &im) {
  // Create zbar scanner
  ImageScanner scanner;

  // Configure scanner for QR only
  scanner.set_config(ZBAR_QRCODE, ZBAR_CFG_ENABLE, 1);
  // canner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);

  // Convert image to grayscale
  cv::Mat imGray;
  cvtColor(im, imGray, cv::COLOR_BGR2GRAY);

  // Wrap image data in a zbar image
  Image image(im.cols, im.rows, "Y800", (uchar *)imGray.data,
              im.cols * im.rows);

  // Scan the image for barcodes and QRCodes
  (void)scanner.scan(image);

  vector<string> result;
  for (Image::SymbolIterator symbol = image.symbol_begin();
       symbol != image.symbol_end(); ++symbol) {
    result.push_back(symbol->get_data());
  }
  return result;
}

std::string ZBarQRDecoder::DecodeSingle(const cv::Mat &image) {
  auto r = Decode(image);
  return r.empty() ? "" : r[0];
}
