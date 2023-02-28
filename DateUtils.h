#pragma once
#include <iostream>


/**
 * Example:
 * 
  Diff dateDiff(1997, 8, 19, 2017, 3, 9);

  Date startDate = dateDiff.startDate();
  Date endDate = dateDiff.endDate();

  printf("Start date: %d/%d/%d \n", startDate.year, startDate.month, startDate.day);
  printf("End date: %d/%d/%d \n", endDate.year, endDate.month, endDate.day);

  Date diffDate = dateDiff.calcDate();
  int diffDays = dateDiff.calcDays();

  printf("Duration between two dates: %d years %d months %d days \n", diffDate.year, diffDate.month, diffDate.day);
  printf("Or: %d days \n", diffDays);
*/

// namespace date {


struct Date {
  int year;
  int month;
  int day;
};

class Diff {
public:
    explicit Diff(int startYear = 1, int startMonth = 1, int startDay = 1, int endYear = 1, int endMonth = 1, int endDay = 1);
    Date startDate() const;
    void setStartDate(int year, int month, int day);
    Date endDate() const;
    void setEndDate(int year, int month, int day);
    int calcLeap() const;
    int calcDays() const;
    Date calcDate() const;

private:
    const int daysPerMonth[24] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}; // in 24 months
    const int elapsedDays[12] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334}; // in 12 months
    int calcLeapInJulianCalendarFromYear(int year) const;
    int calcLeapInGregorianCalendarFromYear(int year) const;
    int calcLeapFromDate(int year, int month, int day) const;
    int calcDaysFromDate(int year, int month, int day) const;
    Date startDate_;
    Date endDate_;
};

// }