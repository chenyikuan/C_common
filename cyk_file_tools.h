#ifndef CYK_FILE_TOOLS_
#define CYK_FILE_TOOLS_

#include <iostream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>    
#include <boost/property_tree/xml_parser.hpp>

namespace cyk_file_tools
//class cyk_file_tools
{
	//public:
	//cyk_file_tools();
	//~cyk_file_tools();
	//static int get_file_contents(const std::string& path_, std::vector<std::string>& contents_str, std::string name_extension = "*", bool if_show = false);
	int get_file_contents(const std::string& path_, std::vector<std::string>& contents_str, std::string name_extension = "*", bool if_show = false);
	/*
	// Get param value from file, in string format.
	static bool get_param_from_file(const std::string& file_name, std::string param_name, std::string& param_val);
	// Get param value from file, in integer format.
	static bool get_param_from_file(const std::string& file_name, std::string param_name, int& param_val);
	// Get param value from file, in float format.
	static bool get_param_from_file(const std::string& file_name, std::string param_name, float& param_val);
	// Get param value from file, in boolean format.
	static bool get_param_from_file(const std::string& file_name, std::string param_name, bool& param_val);
	*/
	template<typename T> bool get_param_from_file(const std::string& file_name, std::string param_name, T& param_val)
	{
		boost::property_tree::ptree m_pt;
		try {
			boost::property_tree::read_xml(file_name, m_pt);
		}
		catch (const boost::property_tree::ptree_error& e) {
			std::cout << std::string("Error reading the config file: ") + e.what() << std::endl;
			return -EXIT_FAILURE;
		}
		param_val = m_pt.get<T>(param_name, T());
		return true;
	}

	template<typename T> bool set_param_from_file(const std::string& file_name, std::string param_name, T& param_val)
	{
		boost::property_tree::ptree m_pt;
		try {
			boost::property_tree::read_xml(file_name, m_pt);
		}
		catch (const boost::property_tree::ptree_error& e) {
			std::cout << std::string("Error reading the config file: ") + e.what() << std::endl;
			return -EXIT_FAILURE;
		}
		m_pt.put<T>(param_name, param_val);
                write_xml(file_name, m_pt);
		printf("OK\n");
		return true;
	}

	template<typename T> bool mkdir(T& path_name)
	{
		boost::filesystem::path p(path_name);
		if (exists(p) && is_directory(p))
		{
			std::cout << "Folder already exits!" << std::endl;
			return true;
		}
		else
		{
			std::cout << "Creating folder: " << path_name << std::endl;
			return create_directory(p);
		}
	}

	bool filename_compare(std::string a, std::string b, std::string prefix);
	int sort_filename(std::vector<std::string>& filenames, std::string prefix = "");

};

#endif

