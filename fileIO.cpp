#ifdef _MSC_VER
#   include <direct.h>
#else
#   include <sys/stat.h>
#endif

#include "fileIO.h"

void MkDir(const std::string& name)
{
#ifdef _MSC_VER
    _mkdir(name.c_str());
#else
    mkdir(name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
}

