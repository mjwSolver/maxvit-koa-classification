export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-100">
        {/* <main className="container mx-auto px-4 py-8">{children}</main> */}
        {children}
      </body>
    </html>
  );
}