#include <string>
#include <stdio.h>

namespace NativeLibrary {
    class NativeClass {
        public:

	    void print_pointer(int l, float* p) {
	    	for (int n=0; n<l; n++) {
			printf("%f \n", p[n]);
		}
	    };
	    
            const std::string& get_property() { return property; }
            void set_property(const std::string& property) { this->property = property; }
            std::string property;
    };
}
