#include <iostream>
#include "cykTools.h"
#include "cyk_file_tools.h"
#include "cykSDH.h"

using namespace std;

int main()
{
	CYK_TOOLS.tic();
	arma::mat a;
	a.randu(1000, 1000);
	SDH fr;
	cout << CYK_TOOLS.random(-10, 10) << endl;
	vector<string> paths;
	cyk_file_tools::get_file_contents(".", paths, "*", true);
	CYK_TOOLS.toc();

	return 0;
}