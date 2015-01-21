#pragma once

#include <cstdlib>
#include <cstdio>
#include <iostream>

class TestGenerator
{
public:
	TestGenerator();
	~TestGenerator();
	static void GenerateTest(std::string dir, int width, int height, int speed);
};

