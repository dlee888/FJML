#ifndef UTIL_INCLUDED
#define UTIL_INCLUDED

#include <iostream>

void progress_bar(int curr, int tot, int bar_width = 69) {
	float progress = (float)curr / tot;
	std::cout << "[";
	int pos = bar_width * progress;
	for (int i = 0; i < bar_width; i++) {
		if (i < pos)
			std::cout << "=";
		else if (i == pos)
			std::cout << ">";
		else
			std::cout << " ";
	}
	std::cout << "] " << int(progress * 100.0) << " %\r";
	std::cout.flush();
}

#endif