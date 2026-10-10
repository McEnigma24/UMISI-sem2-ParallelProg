#include "__preprocessor__.h"



#ifdef BUILD_EXECUTABLE
int main(int argc, char* argv[])
{
    srand(time(NULL));
    // CORE::clear_terminal(); // tests will NOT be VISIBLE with this line
    line("It just works");
    time_stamp("It just works - timestamped");

    

    return 0;
}
#endif