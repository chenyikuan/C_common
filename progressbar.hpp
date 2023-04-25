// The MIT License (MIT)
//
// Copyright (c) 2019 Luigi Pertoldi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//
// ============================================================================
//  ___   ___   ___   __    ___   ____  __   __   ___    __    ___
// | |_) | |_) / / \ / /`_ | |_) | |_  ( (` ( (` | |_)  / /\  | |_)
// |_|   |_| \ \_\_/ \_\_/ |_| \ |_|__ _)_) _)_) |_|_) /_/--\ |_| \_
//
// Very simple progress bar for c++ loops with internal running variable
//
// Author: Luigi Pertoldi
// Created: 3 dic 2016
//
// Notes: The bar must be used when there's no other possible source of output
//        inside the for loop
//

#ifndef __PROGRESSBAR_HPP
#define __PROGRESSBAR_HPP

#include <iostream>
#include <ostream>
#include <string>
#include <stdexcept>
#include <chrono>

class progressbar {

    public:
      // default destructor
      ~progressbar()                             = default;

      // delete everything else
      progressbar           (progressbar const&) = delete;
      progressbar& operator=(progressbar const&) = delete;
      progressbar           (progressbar&&)      = delete;
      progressbar& operator=(progressbar&&)      = delete;

      // default constructor, must call set_niter later
      inline progressbar();
      inline progressbar(int n, bool showbar=true, std::ostream& out=std::cerr);

      // reset bar to use it again
      inline void reset();
     // set number of loop iterations
      inline void set_niter(int iter);
      // chose your style
      inline void set_done_char(const std::string& sym) {done_char = sym;}
      inline void set_todo_char(const std::string& sym) {todo_char = sym;}
      inline void set_opening_bracket_char(const std::string& sym) {opening_bracket_char = sym;}
      inline void set_closing_bracket_char(const std::string& sym) {closing_bracket_char = sym;}
      // to show only the percentage
      inline void show_bar(bool flag = true) {do_show_bar = flag;}
      // set the output stream
      inline void set_output_stream(const std::ostream& stream) {output.rdbuf(stream.rdbuf());}
      // main function
      inline void update();
      inline void update(const std::string& vc);
      inline void end();

    private:
      int progress;
      int n_cycles;
      int last_perc;
      bool do_show_bar;
      bool update_is_called;

      std::string done_char;
      std::string todo_char;
      std::string opening_bracket_char;
      std::string closing_bracket_char;
      std::string tail_chars="";
      std::string verbose_chars="";

      std::ostream& output;
      std::chrono::high_resolution_clock::time_point dbegin0;
      std::chrono::high_resolution_clock::time_point dbegin;
      float last_duration = -1;
};

inline progressbar::progressbar() :
    progress(0),
    n_cycles(0),
    last_perc(0),
    do_show_bar(true),
    update_is_called(false),
    done_char("#"),
    todo_char(" "),
    opening_bracket_char("["),
    closing_bracket_char("]"),
    output(std::cerr) {
        dbegin0 = std::chrono::high_resolution_clock::now();
        dbegin = std::chrono::high_resolution_clock::now();
    }

inline progressbar::progressbar(int n, bool showbar, std::ostream& out) :
    progress(0),
    n_cycles(n),
    last_perc(0),
    do_show_bar(showbar),
    update_is_called(false),
    done_char("#"),
    todo_char(" "),
    opening_bracket_char("["),
    closing_bracket_char("]"),
    output(out) {
        dbegin0 = std::chrono::high_resolution_clock::now();
        dbegin = std::chrono::high_resolution_clock::now();
    }

inline void progressbar::reset() {
    progress = 0,
    update_is_called = false;
    last_perc = 0;
    dbegin0 = std::chrono::high_resolution_clock::now();
    dbegin = std::chrono::high_resolution_clock::now();
    last_duration = -1;
    return;
}

inline void progressbar::set_niter(int niter) {
    if (niter <= 0) throw std::invalid_argument(
        "progressbar::set_niter: number of iterations null or negative");
    n_cycles = niter;
    return;
}

inline void progressbar::update() {

    if (n_cycles == 0) throw std::runtime_error(
            "progressbar::update: number of cycles not set");

    if (!update_is_called) {
        if (do_show_bar == true) {
            output << opening_bracket_char;
            for (int _ = 0; _ < 50; _++) output << todo_char;
            output << closing_bracket_char << " 0%";
        }
        else output << "0%";
    }
    update_is_called = true;

    int perc = 0;

    // compute percentage, if did not change, do nothing and return
    perc = progress*100./(n_cycles-1);
    if (perc < last_perc) return;

    // update percentage each unit
    if (perc == last_perc + 1) {
        // erase the correct  number of characters
        if      (perc <= 10)                output << "\b\b"   << perc << '%';
        else if (perc  > 10 && perc < 100) output << "\b\b\b" << perc << '%';
        else if (perc == 100)               output << "\b\b\b" << perc << '%';
    }
    if (do_show_bar == true) {
        // update bar every ten units
        if (perc % 1 == 0) {
            // erase closing bracket
            output << std::string(closing_bracket_char.size(), '\b');
            // erase trailing percentage characters
            if      (perc  < 10)               output << "\b\b\b";
            else if (perc >= 10 && perc < 100) output << "\b\b\b\b";
            else if (perc == 100)              output << "\b\b\b\b\b";
            // erase tail_chars
            output << std::string(tail_chars.size(), '\b');

            // erase 'todo_char'
            for (int j = 0; j < 50-(perc-1)/2; ++j) {
                output << std::string(todo_char.size(), '\b');
            }

            // add one additional 'done_char'
            if (perc == 0) output << todo_char;
            else           output << done_char;

            // refill with 'todo_char'
            for (int j = 0; j < 50-(perc-1)/2-1; ++j) output << todo_char;

            // readd trailing percentage characters
            output << closing_bracket_char << ' ' << perc << '%';

            tail_chars = " "+std::to_string(progress+1)+"/"+std::to_string(n_cycles);
            tail_chars += " "+verbose_chars;

            std::string remain_str = " [";
            char tm_str[16];
            std::chrono::high_resolution_clock::time_point dend = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> selapsedTime0 = dend - dbegin0;
            float seconds0 = selapsedTime0.count();
            sprintf(tm_str, "%.1f", seconds0);
            std::string tmp = tm_str;
            remain_str += tmp + "S";
            remain_str+=" ETA:";
            auto selapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(dend - dbegin);
            dbegin = dend;
            float seconds = 1.f*selapsedTime.count()/1000;
            seconds = seconds * (n_cycles - progress);
            const int ma_milliseconds = 500;
            if (selapsedTime.count() < ma_milliseconds) {
                float r = 1.f * selapsedTime.count() / ma_milliseconds;
                if (last_duration < 0) {
                    last_duration = seconds;
                } else {
                    last_duration = last_duration * (1.f - r) + seconds * r;
                }
            } else {
                last_duration = seconds;
            }
            if (last_duration > 60) {
                sprintf(tm_str, "%.2f", last_duration/60);
                std::string tmp = tm_str;
                remain_str += tmp + "M]        ";
            } else {
                sprintf(tm_str, "%.1f", last_duration);
                std::string tmp = tm_str;
                remain_str += tmp + "S]        ";
            }
            tail_chars += remain_str;

            output << tail_chars;
        }
    }
    last_perc = perc;
    ++progress;
    output << std::flush;

    return;
}

inline void progressbar::update(const std::string& vc)
{
    verbose_chars = vc;
    update();
}

inline void progressbar::end()
{
    output << std::endl << std::flush;
    return;
}

#endif