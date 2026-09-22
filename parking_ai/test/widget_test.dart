import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:parking_ai/screens/login_screen.dart';

void main() {
  testWidgets('Saioa hasteko pantaila erakusten du', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: LoginScreen()));

    expect(find.text('ParkingAI · Saioa'), findsOneWidget);
    expect(find.text('Saioa hasi'), findsOneWidget);
  });
}
