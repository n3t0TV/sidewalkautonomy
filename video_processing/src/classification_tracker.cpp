#include "classification_tracker.h"

#include <algorithm>

ClassificationTracker::ClassificationTracker(int ncls, int nrecents)
    : nrecents_(nrecents), ncls_(ncls) {}

void ClassificationTracker::Update(int id, const std::vector<float> &scores) {
  if (data_.find(id) == data_.end()) {
    data_.insert({id, Data(ncls_, nrecents_)});
  }
  data_.at(id).Update(scores);
}

void ClassificationTracker::Data::Update(const std::vector<float> &scores) {
  for (int i = 0; i < recents_.size(); ++i) {
    hist_[i] = (n_ * hist_[i] + scores[i]) / (n_ + 1);
    recents_[i] = ((nrecents_ - 1) * recents_[i] + scores[i]) / nrecents_;
  }
  argmax_index_hist_ = static_cast<int>(
      std::max_element(hist_.begin(), hist_.end()) - hist_.begin());
  argmax_index_recent_ = static_cast<int>(
      std::max_element(recents_.begin(), recents_.end()) - recents_.begin());
  ++n_;
}

std::vector<float> ClassificationTracker::GetRecents(int id) const {
  return data_.at(id).recents_;
}

std::vector<float> ClassificationTracker::GetHist(int id) const {
  return data_.at(id).hist_;
}

int ClassificationTracker::GetArgmaxHist(int id) const {
  return data_.at(id).argmax_index_hist_;
}

int ClassificationTracker::GetArgmaxRecent(int id) const {
  return data_.at(id).argmax_index_recent_;
}

std::pair<int, float> ClassificationTracker::GetIdScorePairHist(int id) const {
  int i = data_.at(id).argmax_index_hist_;
  return {i, data_.at(id).hist_[i]};
}

std::pair<int, float> ClassificationTracker::GetIdScorePairRecent(
    int id) const {
  int i = data_.at(id).argmax_index_recent_;
  return {i, data_.at(id).recents_[i]};
}
