#include <iostream>
#include <fstream>


std::ifstream ifs("config.json");
json jf = json::parse(ifs);


