#include "DateUtils.h"

Diff::Diff(int startYear, int startMonth, int startDay, int endYear, int endMonth, int endDay)
{
    setStartDate(startYear, startMonth, startDay);
    setEndDate(endYear, endMonth, endDay);
}

Date Diff::startDate() const
{
    return startDate_;
}

void Diff::setStartDate(int year, int month, int day)
{
    startDate_.year = year;
    startDate_.month = month;
    startDate_.day = day;
}

Date Diff::endDate() const
{
    return endDate_;
}

void Diff::setEndDate(int year, int month, int day)
{
    endDate_.year = year;
    endDate_.month = month;
    endDate_.day = day;
}

int Diff::calcLeap() const
{
    return calcLeapFromDate(endDate_.year, endDate_.month, endDate_.day) - calcLeapFromDate(startDate_.year, startDate_.month, startDate_.day);
}

int Diff::calcDays() const
{
    return calcDaysFromDate(endDate_.year, endDate_.month, endDate_.day) - calcDaysFromDate(startDate_.year, startDate_.month, startDate_.day);
}

Date Diff::calcDate() const
{
    Date diffDate;
    diffDate.year = 0;
    diffDate.month = 0;
    diffDate.day = 0;

    int days = calcDays() - calcLeap(); // make days into common year

    if (days >= 365) {
        diffDate.year = days / 365;
    }
    days -= 365 * diffDate.year;

    int startMonth = 0;
    if (startDate_.year == endDate_.year) {
        startMonth = startDate_.month;
    }
    else if (startDate_.year < endDate_.year) {
        startMonth = endDate_.month;
    }
    if (startMonth > 0) {
        for (int i = startMonth - 1; daysPerMonth[i] <= days; i++) {
        diffDate.month++;
        days -= daysPerMonth[i];
        }
        diffDate.day = days;
    }

    return diffDate;
}

int Diff::calcLeapInJulianCalendarFromYear(int year) const
{
    return year / 4;
}

int Diff::calcLeapInGregorianCalendarFromYear(int year) const
{
    return (year / 4) - (year / 100) + (year / 400);
}

int Diff::calcLeapFromDate(int year, int month, int day) const
{
    // Julian calendar 1582/10/4 -> dropped 10 days -> Gregorian calendar 1582/10/15
    if (year < 1582 || (year == 1582 && month < 10) || (year == 1582 && month == 10 && day < 4)) {
        return calcLeapInJulianCalendarFromYear(year);
    }
    return calcLeapInJulianCalendarFromYear(1582) - 10 + calcLeapInGregorianCalendarFromYear(year) - calcLeapInGregorianCalendarFromYear(1582);
}

int Diff::calcDaysFromDate(int year, int month, int day) const
{
    const int dYear = 365 * (year - 1);
    const int dLeap = calcLeapFromDate(year, month, day);
    const int dMonth = elapsedDays[month - 1];
    return dYear + dLeap + dMonth + day - 1;
}

