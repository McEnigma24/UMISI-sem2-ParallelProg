#include "__preprocessor__.h"

class test
{
    int tab[100];
};

#ifdef BUILD_EXECUTABLE
int main(int argc, char* argv[])
{
    srand(time(NULL));
    // CORE::clear_terminal(); // tests will NOT be VISIBLE with this line
    line("It just works");
    time_stamp("It just works - timestamped");

    CORE::str::split_string("Hello World!", ' ');
    var(CORE::str::to_lower_case("Hello, World!"));

    show_sizeof(test);
    show_sizeof_many(test, 100);

    int num = 123456789;
    double num2 = 1234567.89123;

    cout << CORE::format_number(num) << endl;
    cout << CORE::format_number(num2) << endl;


    for (int i = 0; i < 10; i++)
    {
        var(rand());
        cout.flush();
        sleep(300);
    }

    return 0;
}
#endif