#include "predictor.h"

#include <fstream>

void Predictor::GetStringData(const std::string &filePath, std::string *data) {
  std::ifstream ifs(filePath, std::ios::in | std::ios::binary);
  if (!ifs) {
    data->clear();
    return;
  }
  std::ostringstream oss;
  int len;
  char buf[1024];
  while ((len = ifs.readsome(buf, 1024)) > 0) {
    oss.write(buf, len);
  }
  *data = oss.str();
}
