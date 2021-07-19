//
//  main.cpp
//  GBPI
//
//  Created by ³ÂÒÒ¿í on 15/9/16.
//  Copyright (c) 2015Äê ³ÂÒÒ¿í. All rights reserved.
//

#include <iostream>
#include "cykTools.h"
#include "cyk_file_tools.h"

using namespace std;

int main(int argc, const char * argv[]) {

	vector<string> file_names;
	cyk_file_tools::get_file_contents(".", file_names, "*", true);

	bool bool_val;
	int int_val;
	float float_val;
	double double_val;
	string string_val;
	cyk_file_tools::get_param_from_file("html_test.html", "params.bool", bool_val);
	cyk_file_tools::get_param_from_file("html_test.html", "params.int", int_val);
	cyk_file_tools::get_param_from_file("html_test.html", "params.float", float_val);
	cyk_file_tools::get_param_from_file("html_test.html", "params.double", double_val);
	cyk_file_tools::get_param_from_file("html_test.html", "params.string", string_val);

	cout << bool_val << endl;
	cout << int_val << endl;
	cout << float_val << endl;
	cout << double_val << endl;
	cout << string_val << endl;

	
	CYK_TOOLS.pause();


	return 0;
}


















