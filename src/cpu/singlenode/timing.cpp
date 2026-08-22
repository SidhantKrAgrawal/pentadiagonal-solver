#include "timing.h"
#include <numeric>
#include <cmath>

#ifdef USE_MPI
#  include <mpi.h>
#endif

std::map<std::string, Timing::LoopData> Timing::loops;
int Timing::counter  = 0;
bool Timing::measure = true;

void Timing::startTimer(const std::string &_name) {
  if (!measure) return;
  auto now = clock::now();
  if (loops.size() == 0) counter = 0;
  const std::string fullname = _name;
  if (loops.find(fullname) != loops.end()) {
    loops[fullname].current = now;
  } else {
    loops[fullname] = {counter++, 0.0, now};
  }
}

void Timing::stopTimer(const std::string &_name) {
  if (!measure) return;
  const std::string fullname = _name;
  auto now                   = clock::now();
  loops[fullname].time +=
      std::chrono::duration_cast<std::chrono::duration<double>>(
          now - loops[fullname].current)
          .count();
}

void Timing::reportWithParent(int parent, const std::string &indentation) {
  for (const auto &element : loops) {
    const LoopData &l = element.second;

    std::cout << indentation + element.first + ": "
              << std::to_string(l.time) + " seconds\n";
  }
}

void Timing::reset() {
  loops.clear();
  counter = 0;
}

void Timing::suspend_prof() { measure = false; }
void Timing::continue_prof() { measure = true; }

void Timing::report() { reportWithParent(-1, "  "); }
