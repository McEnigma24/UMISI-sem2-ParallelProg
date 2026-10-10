// clang-format off
#include <benchmark/benchmark.h> // must be first
// clang-format on
#include "__preprocessor__.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <random>

// Trzy strategie tego samego lookupu (klucz -> wartosc), rozniace sie wylacznie ukladem danych.
// Dziedzina klucza jest stala, zmienna niezalezna jest liczba rekordow N.

namespace
{

// constexpr uint32_t KEY_DOMAIN = 72;    // ile roznych wartosci moze przyjac klucz = rozmiar tablicy bezposredniej
constexpr uint32_t KEY_DOMAIN = 1'000;    // ile roznych wartosci moze przyjac klucz = rozmiar tablicy bezposredniej
constexpr size_t QUERIES = 1024;       // dlugosc sekwencji zapytan; potega dwojki, zeby zawijac maska
constexpr size_t QUERIES_MASK = QUERIES - 1;

// Rozmiar dobrany tak, zeby Record mial dokladnie 64 B, czyli jedna linie cache.
struct Payload
{
    uint32_t value;
    uint8_t filler[56];
};

// Wariant 1: klucz siedzi WEWNATRZ duzego rekordu.
struct Record
{
    uint32_t key;
    Payload payload;
};

struct Dataset
{
    std::vector<uint32_t> present;    // klucze faktycznie obecne w strukturze
    std::vector<uint32_t> queries;    // losowa sekwencja zapytan, same trafienia
};

// Staly seed - miedzy uruchomieniami mierzymy dokladnie ten sam uklad danych i te same zapytania.
Dataset make_dataset(uint32_t n)
{
    std::mt19937 rng(12345);

    std::vector<uint32_t> domain(KEY_DOMAIN);
    std::iota(domain.begin(), domain.end(), 0u);
    std::shuffle(domain.begin(), domain.end(), rng);

    Dataset ds;
    ds.present.assign(domain.begin(), domain.begin() + n);

    // Losowa kolejnosc zapytan psuje predykcje skokow - inaczej mierzylibysmy wygrzany wzorzec.
    std::uniform_int_distribution<uint32_t> pick(0, n - 1);
    ds.queries.resize(QUERIES);
    for (auto& q : ds.queries)
        q = ds.present[pick(rng)];

    return ds;
}

double percentile_90(const std::vector<double>& samples)
{
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const size_t idx = static_cast<size_t>(0.9 * static_cast<double>(sorted.size() - 1) + 0.5);
    return sorted[idx];
}

} // namespace

// 1) Zwykla tablica rekordow - match po polu w srodku rekordu.
//    Kazde porownanie 4 B ciagnie cala 64-bajtowa linie cache.
static void BM_lookup_record_scan(benchmark::State& state)
{
    const uint32_t n = static_cast<uint32_t>(state.range(0));
    const Dataset ds = make_dataset(n);

    std::vector<Record> records(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        records[i].key = ds.present[i];
        records[i].payload.value = i * 7 + 1;
    }

    size_t q = 0;
    for (auto _ : state)
    {
        const uint32_t needle = ds.queries[q++ & QUERIES_MASK];

        uint32_t found = 0;
        for (const auto& r : records)
        {
            if (r.key == needle)
            {
                found = r.payload.value;
                break;
            }
        }
        benchmark::DoNotOptimize(found);
    }

    state.SetLabel("AoS: klucz w rekordzie 64 B");
}

// 2) Mala tablica samych kluczy - iterujemy po niej, a pozycja indeksuje tablice obok.
//    Skan dotyka N*4 B, czyli miesci sie w jednej-dwoch liniach cache.
static void BM_lookup_key_array_scan(benchmark::State& state)
{
    const uint32_t n = static_cast<uint32_t>(state.range(0));
    const Dataset ds = make_dataset(n);

    std::vector<uint32_t> keys(n);
    std::vector<Payload> data(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        keys[i] = ds.present[i];
        data[i].value = i * 7 + 1;
    }

    size_t q = 0;
    for (auto _ : state)
    {
        const uint32_t needle = ds.queries[q++ & QUERIES_MASK];

        uint32_t found = 0;
        for (uint32_t i = 0; i < n; ++i)
        {
            if (keys[i] == needle)
            {
                found = data[i].value;
                break;
            }
        }
        benchmark::DoNotOptimize(found);
    }

    state.SetLabel("SoA: ciasna tablica kluczy + tablica obok");
}

// 3) Tablica indeksowana bezposrednio kluczem - zero iteracji, staly koszt niezalezny od N.
//    Placimy za to pamiecia: KEY_DOMAIN slotow zamiast N.
static void BM_lookup_direct_table(benchmark::State& state)
{
    const uint32_t n = static_cast<uint32_t>(state.range(0));
    const Dataset ds = make_dataset(n);

    constexpr uint8_t ABSENT = 0xFF;
    std::array<uint8_t, KEY_DOMAIN> key_to_index{};
    key_to_index.fill(ABSENT);

    std::vector<Payload> data(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        key_to_index[ds.present[i]] = static_cast<uint8_t>(i);
        data[i].value = i * 7 + 1;
    }

    size_t q = 0;
    for (auto _ : state)
    {
        const uint32_t needle = ds.queries[q++ & QUERIES_MASK];

        const uint8_t index = key_to_index[needle];
        uint32_t found = data[index].value;    // nie-const, bo const-ref wersja DoNotOptimize jest deprecated
        benchmark::DoNotOptimize(found);
    }

    state.SetLabel("direct: tablica na KEY_DOMAIN slotow");
}

// Wspolna konfiguracja: N od 8 do 72, powtorzenia z agregatami (mean/median/stddev/cv + p90),
// czas w nanosekundach na jeden lookup. MinTime skrocony, bo inaczej caly sweep trwa minuty.
#define REGISTER_LOOKUP_BENCH(fn)                                                                                      \
    BENCHMARK(fn)                                                                                                      \
        ->DenseRange(8, KEY_DOMAIN, 8)                                                                                 \
        ->Repetitions(10)                                                                                              \
        ->DisplayAggregatesOnly(true)                                                                                  \
        ->ComputeStatistics("p90", percentile_90)                                                                       \
        ->MinTime(0.05)                                                                                                \
        ->Unit(benchmark::kNanosecond)

REGISTER_LOOKUP_BENCH(BM_lookup_record_scan);
REGISTER_LOOKUP_BENCH(BM_lookup_key_array_scan);
REGISTER_LOOKUP_BENCH(BM_lookup_direct_table);

BENCHMARK_MAIN();
