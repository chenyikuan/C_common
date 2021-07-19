#include "cyk_file_tools.h"

#include <iterator>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>    
#include <boost/property_tree/xml_parser.hpp>

using namespace std;
using namespace boost::filesystem;


//cyk_file_tools::cyk_file_tools()
//{
//}
//
//cyk_file_tools::~cyk_file_tools()
//{
//}

int cyk_file_tools::get_file_contents(const std::string& path_, vector<std::string>& contents_str, std::string name_extension, bool if_show)
{
	int count_files;
	vector<path> pathes;
	//vector<string> paths_str;

	path p(path_);   // p reads clearer than argv[1] in the following code

	try
	{
		if (exists(p))    // does p actually exist?
		{
			if (is_regular_file(p))        // is p a regular file?
				cout << p << " size is " << file_size(p) << '\n';

			else if (is_directory(p))      // is p a directory?
			{
				if (if_show)
					cout << p << " is a directory containing:\n";
				copy(directory_iterator(p), directory_iterator(), back_inserter(pathes));
				for (auto iter = pathes.begin(); iter != pathes.end(); iter++){
					if (name_extension == "*" || name_extension == iter->extension().generic_string())
					{
						contents_str.push_back(iter->generic_string());
						if (if_show)
							cout << contents_str.back() << endl;
					}
				}
			}
			else
				cout << p << " exists, but is neither a regular file nor a directory\n";
		}
		else
			cout << p << " does not exist\n";
		return contents_str.size();
	}

	catch (const filesystem_error& ex)
	{
		cout << ex.what() << '\n';
		return -1;
	}
}

bool cyk_file_tools::filename_compare(string a, string b, string prefix)
{
    int length_prefix = prefix.length();
    a = a.substr(length_prefix, a.length() - length_prefix);
    b = b.substr(length_prefix, b.length() - length_prefix);
    if (a.substr(0, 1)=="/")
    	a = a.substr(1, a.length()-1);
    if (b.substr(0, 1)=="/")
    	b = b.substr(1, b.length()-1);
    int num_a = stoi(a);
    int num_b = stoi(b);
    return num_a < num_b;
}

int cyk_file_tools::sort_filename(std::vector<string>& filenames, string prefix)
{
    if (filenames.empty())
    {
    	return 0;
    }
    sort(filenames.begin(), filenames.end(), [prefix](string a, string b)->bool{return filename_compare(a, b, prefix);});
    string tmp_str = filenames[filenames.size()-1];
    tmp_str = tmp_str.substr(prefix.length(), tmp_str.length()-prefix.length());
    if (tmp_str.substr(0, 1)=="/")
    	tmp_str = tmp_str.substr(1, tmp_str.length()-1);
    // std::cout << tmp_str << std::endl;
    return stoi(tmp_str);
}
