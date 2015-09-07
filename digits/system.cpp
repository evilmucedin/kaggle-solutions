#include "system.h"

#ifndef _MSC_VER
#	include <sys/resource.h>
#endif

void SetLowPriority()
{
#ifndef _MSC_VER
    setpriority(PRIO_PROCESS, 0, 20);
#endif
}