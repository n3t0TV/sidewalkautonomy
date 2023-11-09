#ifndef CLASSIFICATION_TRACKER_H_
#define CLASSIFICATION_TRACKER_H_

#include <unordered_map>
#include <vector>

class ClassificationTracker {
  class Data {
    friend class ClassificationTracker;
    std::vector<float> recents_;
    std::vector<float> hist_;
    int argmax_index_hist_;
    int argmax_index_recent_;
    int nrecents_;
    long n_;

   public:
    explicit Data(int ncls, int nrecents)
        : recents_(ncls, 0),
          hist_(ncls, 0),
          argmax_index_hist_{-1},
          argmax_index_recent_{-1},
          nrecents_{nrecents},
          n_{0} {}
    void Update(const std::vector<float> &scores);
  };

 public:
  explicit ClassificationTracker(int ncls, int nrecents = 10);
  void Update(int id, const std::vector<float> &scores);

  bool Exists(int id) const { return data_.find(id) != data_.end(); }
  std::vector<float> GetRecents(int id) const;
  std::vector<float> GetHist(int id) const;
  int GetArgmaxHist(int id) const;
  int GetArgmaxRecent(int id) const;
  std::pair<int, float> GetIdScorePairHist(int id) const;
  std::pair<int, float> GetIdScorePairRecent(int id) const;

 private:
  std::unordered_map<int, Data> data_{};
  int nrecents_;
  int ncls_;
};

#endif  // CLASSIFICATION_TRACKER_H_
